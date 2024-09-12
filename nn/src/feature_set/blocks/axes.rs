use super::FeatureBlock;
use crate::feature_set::axis::Axis;
use shakmaty::{Board, Color, Role, Square};

/// A block of features that computes the index of a piece based on the position, role and color.
/// When built as `product(Horizontal, Vertical)`, the block is the classic 768 feature set
#[derive(Debug)]
pub struct AxesBlock {
    first: Axis,
    second: Option<Axis>,
}

impl AxesBlock {
    pub fn single(axis: Axis) -> Self {
        Self {
            first: axis,
            second: None,
        }
    }

    pub fn product(first: Axis, second: Axis) -> Self {
        Self {
            first,
            second: Some(second),
        }
    }

    /// Computes the index for a given piece. This can be done since the block is piece-independent
    #[inline(always)]
    fn compute_indexes(
        &self,
        piece_square: Square,
        piece_role: Role,
        piece_color: Color,
        perspective: Color,
        features: &mut Vec<u16>,
        offset: u16,
    ) {
        let piece_square = correct_square(piece_square, perspective);
        let piece_role = piece_role as u16 - 1;
        let piece_color = (piece_color != perspective) as u16;

        let mut index: u16 = self.first.index(piece_square);

        if let Some(ref second) = self.second {
            index = index * second.size() as u16 + second.index(piece_square);
        }

        features.push(offset + index * 12 + piece_role * 2 + piece_color);
    }
}

impl FeatureBlock for AxesBlock {
    fn size(&self) -> u16 {
        self.first.size() * self.second.as_ref().map_or(1, |a| a.size()) * 6 * 2
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
        _board: &Board,
        piece_square: Square,
        piece_role: Role,
        piece_color: Color,
        perspective: Color,
        add_feats: &mut Vec<u16>,
        _rem_feats: &mut Vec<u16>,
        offset: u16,
    ) {
        self.compute_indexes(
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
        _board: &Board,
        piece_square: Square,
        piece_role: Role,
        piece_color: Color,
        perspective: Color,
        _add_feats: &mut Vec<u16>,
        rem_feats: &mut Vec<u16>,
        offset: u16,
    ) {
        self.compute_indexes(
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
