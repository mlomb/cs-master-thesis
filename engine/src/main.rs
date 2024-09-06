mod defs;
mod limits;
mod position_stack;
mod pv_table;
mod search;
mod transposition;

use clap::Parser;
use limits::SearchLimits;
use nn::nnue::model::NnueModel;
use search::Search;
use shakmaty::fen::Fen;
use shakmaty::uci::UciMove;
use shakmaty::{CastlingMode, Chess, Color, Position};
use std::cell::RefCell;
use std::io::{self, BufRead};
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

    let model = if let Some(path) = args.nn {
        println!("info string Loading NNUE from {}", path);
        NnueModel::load(&path)
    } else {
        println!("info string Using embedded NNUE");
        NnueModel::from_memory(&include_bytes!("../../models/best.nn").to_vec())
    }
    .expect("Failed to load NNUE model");

    println!("info string NNUE net: {}", model.arch);
    println!("info string NNUE size: {} params", model.params);

    let model = Rc::new(RefCell::new(model));
    let mut search = Search::new(model.clone());
    let mut turn = Color::White;

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
            UciMessage::UciNewGame => {
                // reset the search completely
                // this may be expensive and the object should be reused
                // but this way I'm sure I'm not leaking anything from the previous game
                search = Search::new(model.clone());
                turn = Color::White;
            }
            UciMessage::Position {
                startpos,
                fen,
                moves,
            } => {
                let position = if startpos {
                    Chess::default()
                } else {
                    fen.unwrap()
                        .0
                        .parse::<Fen>()
                        .expect("a valid fen")
                        .into_position(CastlingMode::Standard)
                        .unwrap()
                };

                let moves = moves
                    .iter()
                    .map(|m| m.to_string().parse().expect("a valid uci move"))
                    .collect::<Vec<UciMove>>();

                turn = position.turn();
                search.set_position(position, moves);
            }
            UciMessage::Go {
                time_control,
                search_control,
            } => {
                let best_move =
                    search.go(SearchLimits::from_uci(time_control, search_control, turn));

                println!(
                    "bestmove {}",
                    best_move
                        .expect("a best move")
                        .to_uci(CastlingMode::Standard)
                );
            }
            _ => {}
        }
    }
}
