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

            let square = if perspective == Color::Black {
                // flip square vertically if black is to play, so its below
                square.flip_vertical()
            } else {
                // keep square as is, by default white is below
                square
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

#[cfg(test)]
mod tests {
    use super::FeatureSet;
    use super::*;
    use shakmaty::{fen::Fen, Chess, Position};

    #[test]
    fn sanity_checks() {
        let fen: Fen = "4nrk1/3q1pp1/2n1p1p1/8/1P2Q3/7P/PB1N1PP1/2R3K1 w - - 5 26"
            .parse()
            .unwrap();
        let pos: Chess = fen.into_position(shakmaty::CastlingMode::Standard).unwrap();
        let board_orig = pos.board().clone();
        let mut board_flip = board_orig.clone();
        board_flip.flip_vertical();
        board_flip.swap_colors();

        let basic: Box<dyn FeatureSet> = Box::new(Basic::new());

        let mut feat_orig_white = vec![];
        let mut feat_orig_black = vec![];
        let mut feat_flip_white = vec![];
        let mut feat_flip_black = vec![];

        basic.active_features(&board_orig, Color::White, &mut feat_orig_white);
        basic.active_features(&board_orig, Color::Black, &mut feat_orig_black);
        basic.active_features(&board_flip, Color::White, &mut feat_flip_white);
        basic.active_features(&board_flip, Color::Black, &mut feat_flip_black);

        feat_orig_white.sort();
        feat_orig_black.sort();
        feat_flip_white.sort();
        feat_flip_black.sort();

        assert_eq!(feat_orig_white, feat_flip_black);
        assert_eq!(feat_orig_black, feat_flip_white);
    }
}
