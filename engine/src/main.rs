mod defs;
mod limit;
mod position_stack;
mod pv;
mod search;
mod tt;

use clap::Parser;
use limit::Limit;
use nn::nnue::model::NnueModel;
use search::Search;
use shakmaty::fen::Fen;
use shakmaty::uci::UciMove;
use shakmaty::{CastlingMode, Chess, Position};
use std::cell::RefCell;
use std::fs::File;
use std::io::{self, BufRead, Read};
use std::rc::Rc;
use vampirc_uci::{parse_one, UciMessage};

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

    println!("info string NNUE net: {}", model.arch);
    println!("info string NNUE size: {} params", model.params);

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
                let best_move = search.go(
                    position.clone(),
                    Limit::from_uci(time_control, search_control, position.turn()),
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
