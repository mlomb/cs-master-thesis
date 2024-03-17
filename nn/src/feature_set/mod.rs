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

impl dyn FeatureSet {
    pub fn encoded_size(&self) -> usize {
        self.num_features().div_ceil(64) * 8
    }

    pub fn encode(&self, board: &Board, perspective: Color, write: &mut dyn Write) {
        // extract features from position
        let mut features = vec![];
        self.active_features(board, perspective, &mut features);

        // write into bits of a u64 buffer
        let mut buffer = vec![0u64; self.num_features().div_ceil(64)];

        for feature_index in features.into_iter() {
            assert!(feature_index < self.num_features() as u16);

            let elem_index = (feature_index / 64) as usize;
            let bit_index = (feature_index % 64) as usize;
            buffer[elem_index] |= 1 << bit_index;
        }

        // write buffer into cursor
        write
            .write_all(unsafe {
                std::slice::from_raw_parts(buffer.as_ptr() as *const u8, buffer.len() * 8)
            })
            .unwrap();
    }
}
