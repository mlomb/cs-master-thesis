use super::FeatureBlock;
use shakmaty::{Board, Color, Move, Role, Square};

#[derive(Debug)]
pub struct KingBlock {}

impl KingBlock {
    pub fn new() -> Self {
        Self {}
    }

    /// Computes the index for a given piece. This can be done since the block is piece-independent
    #[inline(always)]
    fn compute_indexes(
        &self,
        board: &Board,
        piece_square: Square,
        piece_role: Role,
        piece_color: Color,
        perspective: Color,
        features: &mut Vec<u16>,
        offset: u16,
    ) {
        let king_sq = board.king_of(perspective).unwrap();
        let king_sq = correct_square(king_sq, perspective);
        let king_file = king_sq.file() as i16;
        let king_rank = king_sq.rank() as i16;

        let piece_square = correct_square(piece_square, perspective);
        let piece_file = piece_square.file() as i16;
        let piece_rank = piece_square.rank() as i16;
        let piece_role = piece_role as u16 - 1;
        let piece_color = (piece_color != perspective) as u16;

        let rel_file = king_file - piece_file + 7;
        let rel_rank = king_rank - piece_rank + 7;
        let index = rel_file * 15 + rel_rank;

        debug_assert!(rel_file >= 0 && rel_file < 15);
        debug_assert!(rel_rank >= 0 && rel_rank < 15);
        debug_assert!(index >= 0 && index < 15 * 15);

        features.push(offset + index as u16 * 12 + piece_role * 2 + piece_color);
    }
}

impl FeatureBlock for KingBlock {
    fn size(&self) -> u16 {
        15 * 15 * 6 * 2
    }

    fn requires_refresh(&self, board: &Board, mov: &Move, turn: Color, perspective: Color) -> bool {
        turn == perspective && board.role_at(mov.from().unwrap()).unwrap() == Role::King
    }

    fn active_features(
        &self,
        board: &Board,
        _turn: Color,
        perspective: Color,
        features: &mut Vec<u16>,
        offset: u16,
    ) {
        for (piece_square, piece) in board.clone().into_iter() {
            self.compute_indexes(
                board,
                piece_square,
                piece.role,
                piece.color,
                perspective,
                features,
                offset,
            );
        }
    }

    fn features_on_add(
        &self,
        board: &Board,
        piece_square: Square,
        piece_role: Role,
        piece_color: Color,
        perspective: Color,
        add_feats: &mut Vec<u16>,
        _rem_feats: &mut Vec<u16>,
        offset: u16,
    ) {
        self.compute_indexes(
            board,
            piece_square,
            piece_role,
            piece_color,
            perspective,
            add_feats, // ←
            offset,
        );
    }

    fn features_on_remove(
        &self,
        board: &Board,
        piece_square: Square,
        piece_role: Role,
        piece_color: Color,
        perspective: Color,
        _add_feats: &mut Vec<u16>,
        rem_feats: &mut Vec<u16>,
        offset: u16,
    ) {
        self.compute_indexes(
            board,
            piece_square,
            piece_role,
            piece_color,
            perspective,
            rem_feats, // ←
            offset,
        );
    }
}

/// Correct square based on perspective
#[inline(always)]
pub fn correct_square(piece_square: Square, perspective: Color) -> Square {
    if perspective == Color::Black {
        // flip square vertically if black is to play, so it is on the bottom side
        piece_square.flip_vertical()
    } else {
        // keep square as is, by default white is below
        piece_square
    }
}
