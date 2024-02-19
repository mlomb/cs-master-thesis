use crate::uci_engine::{Score, UciEngine};

use super::TrainingMethod;
use clap::Args;
use rand::seq::SliceRandom;
use shakmaty::{fen::Fen, Chess, EnPassantMode, Position};
use std::fs::File;
use std::io::{self, Write};

#[derive(Args, Clone)]
pub struct EvalArgs {
    /// UCI engine command to use for evaluation
    #[arg(long, value_name = "engine")]
    engine: String,

    /// Target depth for search
    #[arg(long, value_name = "depth", default_value = "13")]
    depth: usize,
}

pub struct Eval {
    /// Engine instance
    engine: UciEngine,
}

impl Eval {
    pub fn new(args: EvalArgs) -> Self {
        Eval {
            engine: UciEngine::new(&args.engine, args.depth),
        }
    }
}

impl TrainingMethod for Eval {
    fn write_sample(&mut self, file: &mut File, positions: &Vec<Chess>) -> io::Result<()> {
        let mut rng = rand::thread_rng();

        // choose random position
        let position = positions.choose(&mut rng).unwrap().clone();

        // convert to FEN string
        let fen = Fen(position.into_setup(EnPassantMode::Always)).to_string();

        // evaluate and return
        let res = self.engine.evaluate(&fen);

        writeln!(
            file,
            "{},{},{}",
            fen,
            match res.score {
                Score::Cp(c) => format!("{}", c),
                Score::Mate(m) => format!("#{}", m),
            },
            res.best_move
        )
    }
}
