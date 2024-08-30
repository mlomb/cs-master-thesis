use nn::feature_set::FeatureSet;
use shakmaty::Board;
use shakmaty::Chess;
use shakmaty::Color;
use shakmaty::Position;
use std::io::Write;

/// Returns the size of the encoded position in bytes given a feature set
pub fn encoded_size(feature_set: &Box<dyn FeatureSet>) -> usize {
    2 * feature_set.num_features().div_ceil(64) * 8
}

/// Encodes a position (features of both POVs) into a compacted (u64) tensor buffer.
/// First the side to move, then the other
pub fn encode_position(position: &Chess, feature_set: &Box<dyn FeatureSet>, write: &mut dyn Write) {
    let turn = position.turn();
    let board = position.board();

    // encode first side to move, then the other
    encode_side(board, turn, turn, feature_set, write);
    encode_side(board, turn, turn.other(), feature_set, write);
}

/// Encodes a side (features of a single POV) into a compacted (u64) tensor buffer
fn encode_side(
    board: &Board,
    turn: Color,
    perspective: Color,
    feature_set: &Box<dyn FeatureSet>,
    write: &mut dyn Write,
) {
    // extract features from position
    let mut features = vec![];
    feature_set.active_features(board, turn, perspective, &mut features);

    // write into bits of a u64 buffer
    let mut buffer = vec![0u64; feature_set.num_features().div_ceil(64)];

    for feature_index in features.into_iter() {
        debug_assert!(feature_index < feature_set.num_features() as u16);

        let elem_index = (feature_index / 64) as usize;
        let bit_index = (feature_index % 64) as usize;
        buffer[elem_index] |= 1 << bit_index;
    }

    #[cfg(target_endian = "big")]
    assert!(false, "big endian not supported"); // TODO: support big endian ↓

    // write buffer into cursor
    write
        .write_all(unsafe {
            std::slice::from_raw_parts(buffer.as_ptr() as *const u8, buffer.len() * 8)
        })
        .unwrap();
}
