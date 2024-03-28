use super::index_indep::PieceIndependentFeatureSet;
use shakmaty::{Board, Color, Role, Square};

/// The most compact feature set
/// Tuples: <piece_rank, piece_role, piece_color> +
///         <piece_file, piece_role, piece_color>
pub struct HalfCompact;

impl PieceIndependentFeatureSet for HalfCompact {
    fn num_features() -> usize {
        (8 * 6 * 2) * 2 // 192
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
        let piece_square = if perspective == Color::Black {
            // flip square vertically if black is to play, so it is on the bottom side
            piece_square.flip_vertical()
        } else {
            // keep square as is, by default white is below
            piece_square
        };

        let piece_rank = piece_square.rank() as u16;
        let piece_file = piece_square.file() as u16;
        let piece_type = piece_role as u16 - 1;
        let piece_color = (piece_color != perspective) as u16;

        features.push(piece_rank * 12 + piece_type * 2 + piece_color);
        features.push(piece_file * 12 + piece_type * 2 + piece_color + 96);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::feature_set::checks::sanity_checks;

    #[test]
    fn test_sanity_checks() {
        sanity_checks(&HalfCompact {});
    }
}
