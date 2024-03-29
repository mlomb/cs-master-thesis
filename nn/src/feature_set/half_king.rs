use super::index_indep::PieceIndependentFeatureSet;
use shakmaty::{Board, Color, Move, Role, Square};

/// The Half-King-Piece feature set
/// Tuple: <side_king_square, piece_square, piece_role, piece_color>
pub struct HalfKingPiece;

impl PieceIndependentFeatureSet for HalfKingPiece {
    fn num_features() -> usize {
        64 * 64 * 5 * 2 // 40960
    }

    fn requires_refresh(board: &Board, mov: &Move, perspective: Color) -> bool {
        // refresh is needed when the king of the perspective is moved
        mov.role() == Role::King && board.color_at(mov.from().unwrap()).unwrap() == perspective
    }

    #[inline(always)]
    fn compute_indexes(
        board: &Board,
        piece_square: Square,
        piece_role: Role,
        piece_color: Color,
        perspective: Color,
        features: &mut Vec<u16>,
    ) {
        if piece_role == Role::King {
            // skip kings
            return;
        }

        // king location of the perspective
        let king_square = board.king_of(perspective).unwrap();

        let (piece_square, king_square) = if perspective == Color::Black {
            // flip square vertically if black is to play, so it is on the bottom side
            (piece_square.flip_vertical(), king_square.flip_vertical())
        } else {
            // keep square as is, by default white is below
            (piece_square, king_square)
        };

        let side_king_square = king_square as u16;
        let piece_square = piece_square as u16;
        let piece_role = piece_role as u16 - 1;
        let piece_color = (piece_color != perspective) as u16;

        features.push(side_king_square * 640 + piece_square * 10 + piece_role * 2 + piece_color);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::feature_set::checks::sanity_checks;

    #[test]
    fn test_sanity_checks() {
        sanity_checks(&HalfKingPiece {});
    }
}
