pub mod basic;
pub mod halfkp;

use shakmaty::{Board, Color};

/// A set of features for a neural network
pub trait FeatureSet {
    /// Number of features in the set
    fn num_features() -> usize;

    /// Computes the initial features for the given board and perspective (potentially expensive)
    fn init(board: &Board, color: Color, features: &mut Vec<u16>);

    // TODO: accum
}
