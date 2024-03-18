use std::hint::unreachable_unchecked;

use crate::feature_set::FeatureSet;
use shakmaty::{Board, Color, File, Move, Role, Square};

/// The basic feature set
/// Tuple: <piece_square, piece_role>
pub struct Basic;

impl Basic {
    pub fn new() -> Self {
        Basic
    }

    fn make_index(
        piece_square: Square,
        piece_role: Role,
        piece_color: Color,
        perspective: Color,
    ) -> u16 {
        let channel = if piece_color == perspective {
            match piece_role {
                Role::Pawn => 0,
                Role::Knight => 1,
                Role::Bishop => 2,
                Role::Rook => 3,
                Role::Queen => 4,
                Role::King => 5,
            }
        } else {
            match piece_role {
                Role::Pawn => 6,
                Role::Knight => 7,
                Role::Bishop => 8,
                Role::Rook => 9,
                Role::Queen => 10,
                Role::King => 11,
            }
        };

        let square = if perspective == Color::Black {
            // flip square vertically if black is to play, so it is on the bottom side
            piece_square.flip_vertical()
        } else {
            // keep square as is, by default white is below
            piece_square
        };

        channel * 64 + square as u16
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
            features.push(Self::make_index(
                square,
                piece.role,
                piece.color,
                perspective,
            ));
        }
    }

    fn changed_features(
        &self,
        board: &Board,
        _move: &Move,
        perspective: Color,
        add_feats: &mut Vec<u16>,
        rem_feats: &mut Vec<u16>,
    ) {
        let from = _move.from().unwrap();
        let to = _move.to();
        let who_plays = board.color_at(from).unwrap(); // unchecked?

        match _move {
            Move::Normal {
                role,
                from,
                capture,
                to,
                promotion,
            } => {}
            Move::EnPassant { from, to } => {}
            Move::Castle { king, rook } => {
                rem_feats.push(Self::make_index(*king, Role::King, who_plays, perspective));
                rem_feats.push(Self::make_index(*rook, Role::Rook, who_plays, perspective));

                let (king_file, rook_file) = if king < rook {
                    // king side
                    (File::G, File::F)
                } else {
                    // queen side
                    (File::C, File::D)
                };

                add_feats.push(Self::make_index(
                    Square::from_coords(king_file, king.rank()),
                    Role::King,
                    who_plays,
                    perspective,
                ));
                add_feats.push(Self::make_index(
                    Square::from_coords(rook_file, rook.rank()),
                    Role::Rook,
                    who_plays,
                    perspective,
                ));
                return;
            }
            _ => unsafe { unreachable_unchecked() },
        }

        let final_role = _move.promotion().unwrap_or(_move.role());

        add_feats.push(Self::make_index(to, final_role, who_plays, perspective));
        rem_feats.push(Self::make_index(from, _move.role(), who_plays, perspective));

        if _move.is_en_passant() {
            rem_feats.push(Self::make_index(
                Square::from_coords(to.file(), from.rank()),
                Role::Pawn,
                who_plays.other(),
                perspective,
            ));
        } else if let Some(captured) = _move.capture() {
            rem_feats.push(Self::make_index(
                to,
                captured,
                who_plays.other(),
                perspective,
            ));
        }
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
