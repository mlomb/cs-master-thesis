pub mod basic;
//pub mod halfkp;

use shakmaty::{Board, Color, Move};
use std::io::Write;

/// A set of features for a neural network
pub trait FeatureSet {
    /// Number of features in the set
    fn num_features(&self) -> usize;

    /// Whether the given move requires a refresh of the features
    fn requires_refresh(&self, _move: &Move) -> bool;

    /// Computes the initial features for the given board and perspective (potentially expensive)
    fn active_features(&self, board: &Board, perspective: Color, features: &mut Vec<u16>);

    /// Computes the features that have changed with the given move (hopefully fast)
    fn changed_features(
        &self,
        board: &Board,
        _move: &Move,
        perspective: Color,
        added_features: &mut Vec<u16>,
        removed_features: &mut Vec<u16>,
    );
}
