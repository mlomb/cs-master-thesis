use crate::feature_set::FeatureSet;
use shakmaty::{Board, Color, Role};

/// The most common feature set HalfKP
/// Tuple: <our_king_square, piece_square, piece_role (w/o king), piece_color>
pub struct HalfKP;

impl HalfKP {
    pub fn new() -> Self {
        HalfKP
    }
}

impl FeatureSet for HalfKP {
    fn num_features(&self) -> usize {
        64 * 64 * 5 * 2 // 40960
    }

    fn init(&self, board: &Board, active_features: &mut Vec<u16>) {
        active_features.clear();

        let king_square = board.king_of(Color::White).unwrap();

        for (square, piece) in board.clone().into_iter() {
            if piece.role == Role::King {
                continue;
            }

            let p_idx = (piece.role as u16 - 1) * 2 + piece.color as u16;
            let halfkp_idx = square as u16 + (p_idx + king_square as u16 * 10) * 64;

            active_features.push(halfkp_idx);
        }
    }
}
