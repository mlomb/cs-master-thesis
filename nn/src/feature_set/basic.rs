use crate::feature_set::FeatureSet;
use shakmaty::{Board, Color, Move, Role};

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
        64 * 6 * 2 // 768
    }

    fn requires_refresh(&self, _move: &Move) -> bool {
        // this feature set does not require refresh, its very simple
        false
    }

    fn active_features(&self, board: &Board, perspective: Color, features: &mut Vec<u16>) {
        assert!(features.is_empty());

        for (square, piece) in board.clone().into_iter() {
            let channel = if piece.color == perspective {
                match piece.role {
                    Role::Pawn => 0,
                    Role::Knight => 1,
                    Role::Bishop => 2,
                    Role::Rook => 3,
                    Role::Queen => 4,
                    Role::King => 5,
                }
            } else {
                match piece.role {
                    Role::Pawn => 6,
                    Role::Knight => 7,
                    Role::Bishop => 8,
                    Role::Rook => 9,
                    Role::Queen => 10,
                    Role::King => 11,
                }
            };

            features.push(channel * 64 + square as u16);
        }
    }

    fn changed_features(
        &self,
        board: &Board,
        _move: &Move,
        perspective: Color,
        added_features: &mut Vec<u16>,
        removed_features: &mut Vec<u16>,
    ) {
    }
}
