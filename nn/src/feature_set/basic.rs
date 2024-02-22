use crate::feature_set::FeatureSet;
use shakmaty::{Board, Color, Role};

/// The basic feature set
/// Tuple: <piece_square, piece_role>
pub struct Basic;

impl Basic {
    pub fn new() -> Self {
        Basic
    }
}

impl FeatureSet for Basic {
    fn num_features(&self) -> usize {
        64 * 6 // 384
    }

    fn init(&self, board: &Board, color: Color, active_features: &mut Vec<u16>) {
        active_features.clear();

        for (square, piece) in board.clone().into_iter() {
            if piece.color == color {
                let channel = match color {
                    Color::White => match piece.role {
                        Role::Pawn => 0,
                        Role::Knight => 1,
                        Role::Bishop => 2,
                        Role::Rook => 3,
                        Role::Queen => 4,
                        Role::King => 5,
                    },
                    Color::Black => match piece.role {
                        Role::Pawn => 6,
                        Role::Knight => 7,
                        Role::Bishop => 8,
                        Role::Rook => 9,
                        Role::Queen => 10,
                        Role::King => 11,
                    },
                };

                active_features.push(channel * 64 + square as u16);
            }
        }
    }
}
