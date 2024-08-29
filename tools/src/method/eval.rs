use super::{Sample, SampleEncoder};
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

impl SampleEncoder for EvalRead {
    fn x_size(&self, feature_set: &Box<dyn FeatureSet>) -> usize {
        encoded_size(feature_set)
    }

    fn y_size(&self) -> usize {
        4
    }

    fn write_sample(
        &self,
        sample: &Sample,
        write_x: &mut dyn Write,
        write_y: &mut dyn Write,
        feature_set: &Box<dyn FeatureSet>,
    ) {
        encode_position(feature_set, &sample.position, write_x);

        // side to move score
        write_y
            .write_all(&f32::to_le_bytes(sample.score as f32))
            .unwrap();
    }
}
