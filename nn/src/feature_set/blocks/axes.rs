use super::{all::correct_square, FeatureBlock};
use crate::feature_set::axis::Axis;
use shakmaty::{Board, Color, Role, Square};

/// A block of features that computes the index of a piece based on the position (in a single axis), role and color.
#[derive(Debug)]
pub struct AxesBlock {
    axis: Axis,
}

impl AxesBlock {
    pub fn new(axis: Axis) -> Self {
        Self { axis }
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

        features.push(offset + self.axis.index(piece_square) * 12 + piece_role * 2 + piece_color);
    }
}

impl FeatureBlock for AxesBlock {
    fn size(&self) -> u16 {
        self.axis.size() * 6 * 2
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
