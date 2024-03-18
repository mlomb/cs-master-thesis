pub mod eval;
pub mod pqr;

use nn::feature_set::FeatureSet;
use shakmaty::Board;
use shakmaty::Chess;
use shakmaty::Color;
use shakmaty::Position;
use std::io;
use std::io::BufRead;
use std::io::Write;

pub trait WriteSample {
    fn write_sample(&mut self, write: &mut dyn Write, positions: &Vec<Chess>) -> io::Result<()>;
}

pub trait ReadSample {
    fn x_size(&self, feature_set: &Box<dyn FeatureSet>) -> usize;
    fn y_size(&self) -> usize;

    fn read_sample(
        &mut self,
        read: &mut dyn BufRead,
        write_x: &mut dyn Write,
        write_y: &mut dyn Write,
        feature_set: &Box<dyn FeatureSet>,
    );
}

pub fn encoded_size(feature_set: &Box<dyn FeatureSet>) -> usize {
    feature_set.num_features().div_ceil(64) * 8
}

pub fn encode_side(
    feature_set: &Box<dyn FeatureSet>,
    board: &Board,
    perspective: Color,
    write: &mut dyn Write,
) {
    // extract features from position
    let mut features = vec![];
    feature_set.active_features(board, perspective, &mut features);

    // write into bits of a u64 buffer
    let mut buffer = vec![0u64; feature_set.num_features().div_ceil(64)];

    for feature_index in features.into_iter() {
        assert!(feature_index < feature_set.num_features() as u16);

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

pub fn encode_position(feature_set: &Box<dyn FeatureSet>, position: &Chess, write: &mut dyn Write) {
    let board = position.board();

    // encode first side to move, then the other
    encode_side(feature_set, board, position.turn(), write);
    encode_side(feature_set, board, position.turn().other(), write);
}
