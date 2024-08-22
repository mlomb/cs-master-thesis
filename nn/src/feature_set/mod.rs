pub mod build;
mod checks;
mod fs_axes;

use fixedbitset::FixedBitSet;
use shakmaty::{Board, Color};

/// A set of features for a neural network
pub trait FeatureSet {
    /// Number of features in the set
    fn num_features(&self) -> usize;

    /// Computes the initial features for the given board and perspective
    fn active_features(&self, board: &Board, perspective: Color, features: &mut FixedBitSet);
}
