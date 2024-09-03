mod defs;
mod position_stack;
mod pv;
mod search;
mod tt;

use clap::Parser;
use nn::nnue::model::NnueModel;
use search::Search;
use shakmaty::fen::Fen;
use shakmaty::uci::UciMove;
use shakmaty::{CastlingMode, Chess, Color, Position};
use std::cell::RefCell;
use std::fs::File;
use std::io::{self, BufRead, Read};
use std::rc::Rc;
use std::time::Duration;
use vampirc_uci::{parse_one, UciMessage, UciTimeControl};

#[derive(Parser)]
struct Cli {
    /// The neural network file to use (NNUE)
    #[arg(long, value_name = ".nn file")]
    nn: Option<String>,
}

fn main() {
    let args = Cli::parse();

    let nn_file = if let Some(path) = args.nn {
        println!("info string Loading NNUE from {}", path);

        let mut file = File::open(path).unwrap();
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).unwrap();
        buffer
    } else {
        println!("info string Using embedded NNUE");

        include_bytes!("../../models/best.nn").to_vec()
    };

    let model = NnueModel::from_memory(&nn_file).unwrap();
    let mut search = Search::new(Rc::new(RefCell::new(model)));

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
                        name: Some("LimboBot".to_string()),
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

                search.reset_repetition();
                search.record_repetition(&position);

                for m in moves {
                    let uci: UciMove = m.to_string().parse().unwrap();
                    let m = uci.to_move(&position).unwrap();
                    position = position.play(&m).unwrap();

                    search.record_repetition(&position);
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
                if let Some(ref search_control) = search_control {
                    if let Some(opt_depth) = search_control.depth {
                        assert!(opt_depth >= 1);
                        max_depth = Some(opt_depth as i32);
                    }
                }

                let mut max_nodes = None;
                if let Some(ref search_control) = search_control {
                    if let Some(opt_nodes) = search_control.nodes {
                        assert!(opt_nodes >= 1);
                        max_nodes = Some(opt_nodes as usize);
                    }
                }

                let best_move = search.go(
                    position.clone(),
                    max_depth,
                    available_time.map(|t| {
                        // wiggle room to not time out
                        if t < Duration::from_millis(500) {
                            (t - Duration::from_millis(10)).max(Duration::from_millis(10))
                        } else {
                            t - Duration::from_millis(100)
                        }
                    }),
                    max_nodes,
                );

                // TODO: we should investigate why this happens, tho it is very very rare
                let best_move = best_move.or(position.legal_moves().first().cloned());

                println!(
                    "bestmove {}",
                    best_move.unwrap().to_uci(CastlingMode::Standard)
                );
            }
            _ => {}
        }
    }
}
