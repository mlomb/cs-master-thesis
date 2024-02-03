mod chess;
mod encoding;
mod mcts;
mod position;
mod pv;
mod search;

use mcts::{Evaluator, MCTS};
use ndarray::Array2;
use ort::{inputs, Session};
use search::Search;
use shakmaty::fen::Fen;
use shakmaty::uci::Uci;
use shakmaty::{Chess, Color, File, Move, Position, Square};
use std::io::Write;
use std::io::{self, BufRead};
use std::ops::Index;
use std::time::{Duration, Instant};
use vampirc_uci::{
    parse_one, UciMessage, UciMove, UciOptionConfig, UciPiece, UciSquare, UciTimeControl,
};

use crate::encoding::encode_board;

struct NNEvaluator {
    as_color: Color,
    session: Session,
}

impl Evaluator for NNEvaluator {
    fn evaluate(&self, pos: &Chess) -> f32 {
        if pos.is_stalemate() {
            return 0.0;
        }
        if pos.is_checkmate() {
            return 1.0;
        }

        //let mut board = pos.board().clone();
        //if pos.turn() == Color::Black {
        //    board.flip_vertical();
        //    board.swap_colors();
        //}
        //let encoded = encode_board(&board);
        //let mut x = Array2::<i64>::zeros((1, 12));
        //let mut row = x.row_mut(0);
        //
        //for j in 0..12 {
        //    row[j] = encoded[j];
        //}
        //
        //unsafe {
        //    let outputs = self
        //        .session
        //        .run(inputs![x].unwrap_unchecked())
        //        .unwrap_unchecked();
        //    let scores = outputs
        //        .index(0)
        //        .extract_raw_tensor::<f32>()
        //        .unwrap_unchecked()
        //        .1;
        //    let value = -scores[0];
        //
        //    return value;
        //}
        let moves = pos.legal_moves();
        let mut x = Array2::<i64>::zeros((moves.len(), 12));

        assert!(moves.len() > 0, "no legal moves for position: {:?}", pos);

        for i in 0..moves.len() {
            let mut chess_moved = pos.clone();
            chess_moved.play_unchecked(&moves[i]);

            let mut board_moved = chess_moved.board().clone();
            if chess_moved.turn() == Color::Black {
                board_moved.flip_vertical();
                board_moved.swap_colors();
            }

            let mut row = x.row_mut(i);
            let encoded = encode_board(&board_moved);
            for j in 0..12 {
                row[j] = encoded[j];
            }
        }

        let outputs = self.session.run(inputs![x].unwrap()).unwrap();
        let scores = outputs.index(0).extract_raw_tensor::<f32>().unwrap().1;
        return *scores
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
    }
}

fn main() -> ort::Result<()> {
    //ort::init()
    //    .with_execution_providers([CUDAExecutionProvider::default().build()])
    //    .commit()?;

    let session = Session::builder()?
        .with_intra_threads(4)?
        .with_model_from_memory(include_bytes!("../../models/rq-mse-256-tanh-0.535.onnx"))?;

    let mut search = Search::new(session);

    //let mut mcts = MCTS::new(&Chess::default());
    //let mut start1 = Instant::now();
    //let mut start2 = Instant::now();
    //for i in 0..1_000_000 {
    //    mcts.run_iteration();
    //    if i % 10_000 == 0 {
    //        eprintln!("{} at {:?}", i, start2.elapsed());
    //        start2 = Instant::now();
    //    }
    //}
    //let elapsed = start1.elapsed();
    //eprintln!("Elapsed: {:?} {}", elapsed, unsafe { NODES_CREATED });
    //return Ok(());

    // append msg to file
    let mut file = std::fs::OpenOptions::new()
        .append(true)
        .create(true)
        .open("/mnt/c/Users/mlomb/Desktop/Tesis/cs-master-thesis/bot/log.txt")
        .unwrap();

    //let mut evaluator = NNEvaluator {
    //    as_color: Color::White,
    //    session,
    //};
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

                let available_time = match time_control {
                    None => Duration::from_secs(10),
                    Some(UciTimeControl::Infinite) => Duration::MAX,
                    Some(UciTimeControl::Ponder) => unimplemented!(""),
                    Some(UciTimeControl::MoveTime(fixed_time)) => fixed_time.to_std().unwrap(), // movetime X (ms)
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
                        Duration::from_millis(ms as u64)
                    }
                };

                let mut depth = 10;
                if let Some(search_control) = search_control {
                    if let Some(opt_depth) = search_control.depth {
                        assert!(opt_depth >= 1);
                        depth = opt_depth as i32;
                        depth = depth.min(7);
                    }
                }

                search.go(&position, depth);

                //let start = Instant::now();
                //let mut vis = 0;
                //let (score, mv) = search::negamax(
                //    position.clone(),
                //    depth,
                //    f32::NEG_INFINITY,
                //    f32::INFINITY,
                //    &session,
                //    &mut vis,
                //);
                //let elapsed = start.elapsed();
                //eprintln!(
                //    "**** {} {} in {} -- {} nodes",
                //    mv.clone().unwrap(),
                //    score,
                //    elapsed.as_millis(),
                //    vis
                //);
                //println!("{}", UciMessage::best_move(move_to_uci(mv.unwrap())));

                //evaluator.as_color = position.turn();
                //let start = Instant::now();
                //let mut mcts = MCTS::new(&position, &evaluator);
                //let mut its = 0;
                //loop {
                //    mcts.run_iteration();
                //    its += 1;

                //    if its % 10 == 0 {
                //        // check break conditions
                //        if start.elapsed() >= available_time - Duration::from_millis(5) {
                //            break;
                //        }
                //    }
                //}
                //eprintln!("**** its={} time={}ms", its, start.elapsed().as_millis());
                ////eprintln!("distr={:?}", mcts.get_action_distribution());

                //let best_move = mcts.get_best_action();
                //println!("{}", UciMessage::best_move(move_to_uci(best_move.unwrap())));
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
