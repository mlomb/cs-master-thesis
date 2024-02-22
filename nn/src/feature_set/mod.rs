pub mod basic;
pub mod halfkp;

use shakmaty::{Board, Chess, Color};

/// A set of features for a neural network
pub trait FeatureSet {
    /// Number of features in the set
    fn num_features(&self) -> usize;

    /// Computes the initial features for the given board and perspective (potentially expensive)
    fn init(&self, board: &Board, color: Color, features: &mut Vec<u16>);

    // TODO: accum
}

impl dyn FeatureSet {
    pub fn write_inputs(&self, position: &Chess, buffer: &mut [u64]) {
        //
    }
}
