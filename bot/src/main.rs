mod encoding;
mod search;

use ort::Session;
use shakmaty::fen::Fen;
use shakmaty::uci::Uci;
use shakmaty::{Chess, File, Move, Position, Square};
use std::io::Write;
use std::io::{self, BufRead};
use vampirc_uci::{parse_one, UciMessage, UciMove, UciOptionConfig, UciPiece, UciSquare};

fn main() -> ort::Result<()> {
    let session = Session::builder()?
        .with_intra_threads(1)?
        .with_model_from_memory(include_bytes!("../../models/model-0682-1.751.onnx"))?;

    // append msg to file
    let mut file = std::fs::OpenOptions::new()
        .append(true)
        .create(true)
        .open("/mnt/c/Users/mlomb/Desktop/Tesis/cs-master-thesis/bot/log.txt")
        .unwrap();

    let mut position: Chess = Chess::default();

    for line in io::stdin().lock().lines() {
        let msg: UciMessage = parse_one(&line.unwrap().trim());

        file.write_all(format!("{}\n", msg).as_bytes()).unwrap();

        match msg {
            UciMessage::Uci => {
                println!(
                    "{}",
                    UciMessage::Id {
                        name: Some("thesisbot".to_string()),
                        author: Some("mlomb".to_string())
                    }
                );
                println!(
                    "{}",
                    UciMessage::Option(UciOptionConfig::String {
                        name: "Dummy".to_string(),
                        default: None,
                    })
                );
                println!("{}", UciMessage::UciOk);
            }
            UciMessage::IsReady => {
                println!("{}", UciMessage::ReadyOk);
            }
            UciMessage::Quit => {
                break;
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
                // write controls
                file.write_all(format!("{:?}\n", time_control).as_bytes())
                    .unwrap();
                file.write_all(format!("{:?}\n", search_control).as_bytes())
                    .unwrap();

                let (score, mv) = search::negamax(
                    position.clone(),
                    5,
                    f32::NEG_INFINITY,
                    f32::INFINITY,
                    &session,
                );

                eprintln!("Search result: {} {}", mv, score);
                println!("{}", UciMessage::best_move(move_to_uci(mv)));
            }
            _ => {}
        }

        // println!("Received message: {}", msg);
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
