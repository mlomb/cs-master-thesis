use super::FeatureSet;
use shakmaty::{Board, Color, Move, Role, Square};

/// The most compact feature set
/// Tuples: <piece_rank, piece_role, piece_color> +
///         <piece_file, piece_role, piece_color>
pub struct HalfCompact;

impl HalfCompact {
    #[inline(always)]
    fn compute_indexes(
        piece_square: Square,
        piece_role: Role,
        piece_color: Color,
        perspective: Color,
    ) -> (u16, u16) {
        let piece_square = if perspective == Color::Black {
            // flip square vertically if black is to play, so it is on the bottom side
            piece_square.flip_vertical()
        } else {
            // keep square as is, by default white is below
            piece_square
        };

        let piece_rank = piece_square.rank() as u16;
        let piece_file = piece_square.file() as u16;
        let piece_role = piece_role as u16 - 1;
        let piece_color = (piece_color != perspective) as u16;

        (
            piece_file * 12 + piece_role * 2 + piece_color,
            piece_rank * 12 + piece_role * 2 + piece_color + 96,
        )
    }
}

impl FeatureSet for HalfCompact {
    fn num_features(&self) -> usize {
        (8 * 6 * 2) * 2 // 192
    }

    fn requires_refresh(&self, _: &Board, _: &Move, _: Color) -> bool {
        // Always use refresh
        // Not being piece independent is annoying to code ;)
        // And its not a good feature set anyway
        true
    }

    fn active_features(&self, board: &Board, perspective: Color, features: &mut Vec<u16>) {
        let mut indexes = [false; 192];

        for (square, piece) in board.clone().into_iter() {
            let (ifile, irank) =
                Self::compute_indexes(square, piece.role, piece.color, perspective);

            indexes[ifile as usize] = true;
            indexes[irank as usize] = true;
        }

        debug_assert!(features.is_empty());

        for (index, is_present) in indexes.into_iter().enumerate() {
            if is_present {
                features.push(index as u16);
            }
        }
    }

    /// Unused
    fn changed_features(&self, _: &Board, _: &Move, _: Color, _: &mut Vec<u16>, _: &mut Vec<u16>) {
        unimplemented!()
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
