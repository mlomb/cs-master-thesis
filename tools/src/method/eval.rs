use super::Sample;
use crate::encode::{encode_position, encoded_size};
use clap::Args;
use nn::feature_set::FeatureSet;
use shakmaty::{fen::Fen, uci::UciMove, CastlingMode, Chess, Position};
use std::io::{BufRead, Write};

#[derive(Args, Clone)]
pub struct EvalArgs {
    /// UCI engine command to use for evaluation
    #[arg(long, value_name = "engine")]
    engine: String,

    /// Target depth for search
    #[arg(long, value_name = "depth", default_value = "10")]
    depth: usize,
}

/// Score of a position, given by the engine
#[derive(Debug)]
pub enum Score {
    /// Centipawn
    Cp(i32),

    /// Mate/Mated in n
    Mate(),
}

pub struct EvalRead;

impl Sample for EvalRead {
    fn x_size(&self, feature_set: &Box<dyn FeatureSet>) -> usize {
        encoded_size(feature_set)
    }

    fn y_size(&self) -> usize {
        4
    }

    fn read_sample(
        &self,
        read: &mut dyn BufRead,
        write_x: &mut dyn Write,
        write_y: &mut dyn Write,
        feature_set: &Box<dyn FeatureSet>,
    ) {
        let mut fen_bytes = Vec::with_capacity(128);
        let mut score_bytes = Vec::with_capacity(16);
        let mut bestmove_bytes = Vec::with_capacity(16);

        read.read_until(b',', &mut fen_bytes).unwrap();
        read.read_until(b',', &mut score_bytes).unwrap();
        read.read_until(b'\n', &mut bestmove_bytes).unwrap();

        if fen_bytes.is_empty() {
            // Note: this can happen if the file has a mate score in the last line
            return;
        }

        // remove trailing comma & newline
        fen_bytes.pop();
        score_bytes.pop();
        if bestmove_bytes.last() == Some(&b'\n') {
            // remove trailing newline
            // it may not be present in the last line
            bestmove_bytes.pop();
        }

        let score_str = String::from_utf8_lossy(&score_bytes);
        let score = if let Ok(score) = score_str.parse::<i32>() {
            Score::Cp(score)
        } else {
            // Score::Mate(score_str[1..].parse::<i32>().unwrap())
            Score::Mate()
        };

        if let Score::Cp(cp_score) = score {
            let fen = Fen::from_ascii(fen_bytes.as_slice()).unwrap();
            let position: Chess = fen.into_position(CastlingMode::Standard).unwrap();

            let best_move_uci: UciMove = UciMove::from_ascii(bestmove_bytes.as_slice()).unwrap();
            let best_move = best_move_uci.to_move(&position).unwrap();

            if best_move.is_capture() || position.is_check() {
                // skip capture and check positions
                return;
            }

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
