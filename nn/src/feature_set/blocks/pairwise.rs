use super::FeatureBlock;
use crate::feature_set::axis::Axis;
use shakmaty::{Board, Color, Move, Role, Square};

#[derive(Debug)]
pub struct PairwiseBlock {
    axis: Axis,
}

impl FeatureBlock for PairwiseBlock {
    fn size(&self) -> u16 {
        5
    }

    fn requires_refresh(&self, board: &Board, mov: &Move, turn: Color, perspective: Color) -> bool {
        todo!()
    }

    fn active_features(
        &self,
        board: &Board,
        turn: Color,
        perspective: Color,
        features: &mut Vec<u16>,
        offset: u16,
    ) {
        todo!()
    }

    fn features_on_add(
        &self,
        board: &Board,
        piece_square: Square,
        piece_role: Role,
        piece_color: Color,
        perspective: Color,
        add_feats: &mut Vec<u16>,
        rem_feats: &mut Vec<u16>,
        offset: u16,
    ) {
        todo!()
    }

    fn features_on_remove(
        &self,
        board: &Board,
        piece_square: Square,
        piece_role: Role,
        piece_color: Color,
        perspective: Color,
        add_feats: &mut Vec<u16>,
        rem_feats: &mut Vec<u16>,
        offset: u16,
    ) {
        todo!()
    }
}
