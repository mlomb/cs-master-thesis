pub mod half_compact;
pub mod half_king;
pub mod half_piece;

mod checks;
mod index_indep;

use shakmaty::{Board, Color, Move};

/// A set of features for a neural network
pub trait FeatureSet {
    /// Number of features in the set
    fn num_features(&self) -> usize;

    /// Whether the given move requires a refresh of the features
    fn requires_refresh(&self, board: &Board, mov: &Move, perspective: Color) -> bool;

    /// Computes the initial features for the given board and perspective (potentially expensive)
    fn active_features(&self, board: &Board, perspective: Color, features: &mut Vec<u16>);

    /// Computes the features that have changed with the given move (hopefully fast)
    fn changed_features(
        &self,
        board: &Board,
        mov: &Move,
        perspective: Color,
        added_features: &mut Vec<u16>,
        removed_features: &mut Vec<u16>,
    );
}
