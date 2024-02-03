mod encoding;
mod pv;
mod search;

use ort::Session;
use search::Search;
use shakmaty::fen::Fen;
use shakmaty::uci::Uci;
use shakmaty::{Chess, Color, File, Move, Position, Square};
use std::io::{self, BufRead};
use std::time::Duration;
use vampirc_uci::{parse_one, UciMessage, UciMove, UciPiece, UciSquare, UciTimeControl};

fn main() -> ort::Result<()> {
    //ort::init()
    //    .with_execution_providers([CUDAExecutionProvider::default().build()])
    //    .commit()?;

    let session = Session::builder()?
        .with_intra_threads(4)?
        .with_model_from_memory(include_bytes!("../../models/rq-mse-256-0.470.onnx"))?;

    let mut search = Search::new(session);

    let mut position: Chess = Chess::default();

    for line in io::stdin().lock().lines() {
        let msg: UciMessage = parse_one(&line.unwrap().trim());

        match msg {
            UciMessage::IsReady => println!("{}", UciMessage::ReadyOk),
            UciMessage::Quit => break,
            UciMessage::Uci => {
                println!(
                    "{}",
                    UciMessage::Id {
                        name: Some("thesisbot".to_string()),
                        author: Some("mlomb".to_string())
                    }
                );
                println!("{}", UciMessage::UciOk);
            }
            UciMessage::Position {
                startpos,
                fen,
                moves,
            } => {
                if startpos {
                    position = Chess::default();
                } else {
                    let fen: Fen = fen.unwrap().0.parse().unwrap();
                    position = fen.into_position(shakmaty::CastlingMode::Standard).unwrap();
                }

                for m in moves {
                    let uci: Uci = m.to_string().parse().unwrap();
                    let m = uci.to_move(&position).unwrap();
                    position = position.play(&m).unwrap();
                }
            }
            UciMessage::Go {
                time_control,
                search_control,
            } => {
                let available_time = match time_control {
                    None => None, // infinite
                    Some(UciTimeControl::Infinite) => None,
                    Some(UciTimeControl::Ponder) => unimplemented!(""),
                    Some(UciTimeControl::MoveTime(fixed_time)) => fixed_time.to_std().ok(), // movetime X (ms)
                    Some(UciTimeControl::TimeLeft {
                        white_time,
                        black_time,
                        white_increment,
                        black_increment,
                        moves_to_go: _,
                    }) => {
                        let white_time = white_time.map(|x| x.num_milliseconds()).unwrap_or(0);
                        let black_time = black_time.map(|x| x.num_milliseconds()).unwrap_or(0);
                        let white_incr = white_increment.map(|x| x.num_milliseconds()).unwrap_or(0);
                        let black_incr = black_increment.map(|x| x.num_milliseconds()).unwrap_or(0);

                        let (my_time, my_incr) = if position.turn() == Color::White {
                            (white_time, white_incr)
                        } else {
                            (black_time, black_incr)
                        };

                        let ms = my_incr as f32 + 0.02 * my_time as f32;
                        Some(Duration::from_millis(ms as u64))
                    }
                };

                let mut max_depth = None;
                if let Some(search_control) = search_control {
                    if let Some(opt_depth) = search_control.depth {
                        assert!(opt_depth >= 1);
                        max_depth = Some(opt_depth as i32);
                    }
                }

                let best_move = search.go(
                    &position,
                    max_depth,
                    available_time.map(|t| t - Duration::from_millis(100)),
                );

                println!("{}", UciMessage::best_move(move_to_uci(best_move.unwrap())));
            }
            _ => {}
        }
    }

    Ok(())
}

pub fn move_to_uci(mv: Move) -> UciMove {
    UciMove {
        from: square_to_uci(mv.from().unwrap()),
        to: square_to_uci(mv.to()),
        promotion: mv.promotion().map(|p| match p {
            shakmaty::Role::Knight => UciPiece::Knight,
            shakmaty::Role::Bishop => UciPiece::Bishop,
            shakmaty::Role::Rook => UciPiece::Rook,
            shakmaty::Role::Queen => UciPiece::Queen,
            _ => panic!("Invalid promotion"),
        }),
    }
}

pub fn square_to_uci(sq: Square) -> UciSquare {
    UciSquare {
        file: match sq.file() {
            File::A => 'a',
            File::B => 'b',
            File::C => 'c',
            File::D => 'd',
            File::E => 'e',
            File::F => 'f',
            File::G => 'g',
            File::H => 'h',
        },
        rank: sq.rank() as u8 + 1,
    }
}
