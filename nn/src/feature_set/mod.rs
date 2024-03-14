pub mod basic;
pub mod halfkp;

use shakmaty::{Board, Color};
use std::io::Write;

/// A set of features for a neural network
pub trait FeatureSet {
    /// Number of features in the set
    fn num_features(&self) -> usize;

    /// Computes the initial features for the given board and perspective (potentially expensive)
    fn init(&self, board: &Board, features: &mut Vec<u16>);

    // TODO: accum
}

impl dyn FeatureSet {
    pub fn encoded_size(&self) -> usize {
        self.num_features().div_ceil(64) * 8
    }

    pub fn encode(&self, board: &Board, write: &mut dyn Write) {
        // extract features from position
        let mut features = vec![];
        self.init(board, &mut features);

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
