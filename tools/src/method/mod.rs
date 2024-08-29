pub mod eval;
pub mod pqr;

use nn::feature_set::FeatureSet;
use shakmaty::Chess;
use shakmaty::Move;
use std::io::Write;

#[derive(Debug)]
pub struct Sample {
    /// The position
    pub position: Chess,
    /// Best move in the position, chosen by the engine
    pub bestmove: Move,
    /// Score of the position from the POV of the side to move, in centipawns
    pub score: i32,
}

pub trait SampleEncoder {
    /// Size of the input tensor
    fn x_size(&self, feature_set: &Box<dyn FeatureSet>) -> usize;

    /// Size of the target tensor
    fn y_size(&self) -> usize;

    /// Encodes a sample into the input and output tensors using the given feature set.
    /// It may not write the sample, not write anything (e.g. skipping capture positions).
    fn write_sample(
        &self,
        sample: &Sample,
        write_x: &mut dyn Write,
        write_y: &mut dyn Write,
        feature_set: &Box<dyn FeatureSet>,
    );
}
