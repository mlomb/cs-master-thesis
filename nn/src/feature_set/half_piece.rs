use super::index_indep::PieceIndependentFeatureSet;
use shakmaty::{Board, Color, Role, Square};

/// The Half-Piece feature set
/// Tuple: <piece_square, piece_role, piece_color>
pub struct HalfPiece;

impl PieceIndependentFeatureSet for HalfPiece {
    fn num_features() -> usize {
        64 * 6 * 2 // 768
    }

    #[inline(always)]
    fn compute_indexes(
        _board: &Board,
        piece_square: Square,
        piece_role: Role,
        piece_color: Color,
        perspective: Color,
        features: &mut Vec<u16>,
    ) {
        let piece_square = if perspective == Color::Black {
            // flip square vertically if black is to play, so it is on the bottom side
            piece_square.flip_vertical()
        } else {
            // keep square as is, by default white is below
            piece_square
        };

        let piece_square = piece_square as u16;
        let piece_role = piece_role as u16 - 1;
        let piece_color = (piece_color != perspective) as u16;

        // TODO: move to new format, keeping old one to preserve current net
        //features.push(piece_square * 12 + piece_role * 2 + piece_color);
        features.push((piece_role + 6 * piece_color) * 64 + piece_square);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::feature_set::checks::sanity_checks;

    #[test]
    fn test_sanity_checks() {
        sanity_checks(&HalfPiece {});
    }
}
