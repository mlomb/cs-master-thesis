use super::FeatureSet;
use shakmaty::{Board, Color, File, Move, Role, Square};

pub trait PieceIndependentFeatureSet {
    fn num_features() -> usize;

    fn compute_indexes(
        board: &Board,
        piece_square: Square,
        piece_role: Role,
        piece_color: Color,
        perspective: Color,
        features: &mut Vec<u16>,
    );
}

impl<T: PieceIndependentFeatureSet> FeatureSet for T {
    fn num_features(&self) -> usize {
        Self::num_features()
    }

    fn requires_refresh(&self, _move: &Move) -> bool {
        // :)
        false
    }

    fn active_features(&self, board: &Board, perspective: Color, features: &mut Vec<u16>) {
        debug_assert!(features.is_empty());

        for (square, piece) in board.clone().into_iter() {
            Self::compute_indexes(
                board,
                square,
                piece.role,
                piece.color,
                perspective,
                features,
            );
        }
    }

    fn changed_features(
        &self,
        board: &Board,
        m: &Move,
        perspective: Color,
        add_feats: &mut Vec<u16>,
        rem_feats: &mut Vec<u16>,
    ) {
        let from = m.from().unwrap();
        let to = m.to();
        let who_plays = board.color_at(from).unwrap(); // unchecked?

        match m {
            Move::Normal { from, to, .. } | Move::EnPassant { from, to } => {
                let final_role = m.promotion().unwrap_or(m.role());

                Self::compute_indexes(&board, *to, final_role, who_plays, perspective, add_feats);
                Self::compute_indexes(&board, *from, m.role(), who_plays, perspective, rem_feats);
            }
            Move::Castle { king, rook } => {
                Self::compute_indexes(&board, *king, Role::King, who_plays, perspective, rem_feats);
                Self::compute_indexes(&board, *rook, Role::Rook, who_plays, perspective, rem_feats);

                let (king_file, rook_file) = if king < rook {
                    // king side
                    (File::G, File::F)
                } else {
                    // queen side
                    (File::C, File::D)
                };

                Self::compute_indexes(
                    &board,
                    Square::from_coords(king_file, king.rank()),
                    Role::King,
                    who_plays,
                    perspective,
                    add_feats,
                );
                Self::compute_indexes(
                    &board,
                    Square::from_coords(rook_file, rook.rank()),
                    Role::Rook,
                    who_plays,
                    perspective,
                    add_feats,
                );
                return;
            }
            _ => unreachable!(),
        }

        if m.is_en_passant() {
            Self::compute_indexes(
                &board,
                Square::from_coords(to.file(), from.rank()),
                Role::Pawn,
                who_plays.other(),
                perspective,
                rem_feats,
            );
        } else if let Some(captured) = m.capture() {
            Self::compute_indexes(
                &board,
                to,
                captured,
                who_plays.other(),
                perspective,
                rem_feats,
            );
        }
    }
}
