use crate::feature_set::FeatureSet;
use shakmaty::{Board, Color, Role};

/// The most common feature set HalfKP
/// Tuple: <our_king_square, piece_square, piece_role (w/o king), piece_color>
pub struct HalfKP;

impl FeatureSet for HalfKP {
    fn num_features() -> usize {
        64 * 64 * 5 * 2 // 40960
    }

    fn init(board: &Board, color: Color, active_features: &mut Vec<u16>) {
        active_features.clear();

        let king_square = board.king_of(color).unwrap();

        for (square, piece) in board.clone().into_iter() {
            if piece.role == Role::King {
                continue;
            }

            let p_idx = piece.role as u16 * 2 + piece.color as u16;
            let halfkp_idx = square as u16 + (p_idx + king_square as u16 * 10) * 64;

            active_features.push(halfkp_idx);
        }
    }
}
