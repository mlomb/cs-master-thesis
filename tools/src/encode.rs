use nn::feature_set::FeatureSet;
use shakmaty::Board;
use shakmaty::Chess;
use shakmaty::Color;
use shakmaty::Position;
use std::io::Write;

pub fn encoded_size(feature_set: &Box<dyn FeatureSet>) -> usize {
    2 * feature_set.num_features().div_ceil(64) * 8
}

pub fn encode_side(
    feature_set: &Box<dyn FeatureSet>,
    board: &Board,
    turn: Color,
    perspective: Color,
    write: &mut dyn Write,
) {
    // extract features from position
    let mut features = vec![];
    feature_set.active_features(board, turn, perspective, &mut features);

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
    let turn = position.turn();
    let board = position.board();

    // encode first side to move, then the other
    encode_side(feature_set, board, turn, turn, write);
    encode_side(feature_set, board, turn, turn.other(), write);
}
