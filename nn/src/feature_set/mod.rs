pub mod build;
mod checks;
mod fs_axes;

use shakmaty::{Board, Color, Move};

/// A set of features for a neural network
pub trait FeatureSet {
    /// Number of features in the set
    fn num_features(&self) -> usize;

    /// Whether the given move requires a refresh of the features
    fn requires_refresh(&self, board: &Board, mov: &Move, turn: Color, perspective: Color) -> bool;

    /// Computes the initial features for the given board and perspective (potentially slow)
    fn active_features(
        &self,
        board: &Board,
        turn: Color,
        perspective: Color,
        features: &mut Vec<u16>,
    );

    /// Computes the features that have changed with the given move (hopefully fast)
    fn changed_features(
        &self,
        board: &Board,
        mov: &Move,
        turn: Color,
        perspective: Color,
        added_features: &mut Vec<u16>,
        removed_features: &mut Vec<u16>,
    );
}
