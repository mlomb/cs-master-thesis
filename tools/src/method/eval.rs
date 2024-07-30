use super::{encode_position, encoded_size, ReadSample, WriteSample};
use crate::uci_engine::{Score, UciEngine};
use clap::Args;
use nn::feature_set::FeatureSet;
use rand::seq::SliceRandom;
use shakmaty::{fen::Fen, CastlingMode, Chess, EnPassantMode, Position};
use std::io::{self, BufRead, Write};

#[derive(Args, Clone)]
pub struct EvalArgs {
    /// UCI engine command to use for evaluation
    #[arg(long, value_name = "engine")]
    engine: String,

    /// Target depth for search
    #[arg(long, value_name = "depth", default_value = "10")]
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

impl WriteSample for Eval {
    fn write_sample(&mut self, write: &mut dyn Write, positions: &Vec<Chess>) -> io::Result<()> {
        let mut rng = rand::thread_rng();

        // choose random position
        let position = positions.choose(&mut rng).unwrap().clone();

        // convert to FEN string
        let fen = Fen(position.into_setup(EnPassantMode::Always)).to_string();

        // evaluate and return
        let res = self.engine.evaluate(&fen);

        writeln!(
            write,
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

pub struct EvalRead;

impl ReadSample for EvalRead {
    fn x_size(&self, feature_set: &Box<dyn FeatureSet>) -> usize {
        encoded_size(feature_set)
    }

    fn y_size(&self) -> usize {
        4
    }

    fn read_sample(
        &mut self,
        read: &mut dyn BufRead,
        write_x: &mut dyn Write,
        write_y: &mut dyn Write,
        feature_set: &Box<dyn FeatureSet>,
    ) {
        let mut fen_bytes = Vec::with_capacity(128);
        let mut score_bytes = Vec::with_capacity(16);
        let mut bestmove_bytes = Vec::with_capacity(16); // unused

        read.read_until(b',', &mut fen_bytes).unwrap();
        read.read_until(b',', &mut score_bytes).unwrap();
        read.read_until(b'\n', &mut bestmove_bytes).unwrap();

        if fen_bytes.is_empty() {
            // Note: this can happen if the file has a mate score in the last line
            return;
        }

        // remove trailing comma
        fen_bytes.pop();
        score_bytes.pop();

        let score_str = String::from_utf8_lossy(&score_bytes);
        let score = if let Ok(score) = score_str.parse::<i32>() {
            Score::Cp(score)
        } else {
            Score::Mate(score_str[1..].parse::<i32>().unwrap())
        };

        if let Score::Cp(cp_score) = score {
            let fen = Fen::from_ascii(fen_bytes.as_slice()).unwrap();
            let position: Chess = fen.into_position(CastlingMode::Standard).unwrap();

            encode_position(feature_set, &position, write_x);

            // side to move score
            write_y
                .write_all(&f32::to_le_bytes(cp_score as f32))
                .unwrap();
        } else {
            // else skip mate scores
        }
    }
}
