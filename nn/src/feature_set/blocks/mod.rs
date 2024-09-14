pub mod axes;
pub mod king;
pub mod mobility;
pub mod pairwise;

use crate::feature_set::blocks::axes::AxesBlock;
use crate::feature_set::blocks::pairwise::PairwiseBlock;
use enum_dispatch::enum_dispatch;
use king::KingBlock;
use mobility::MobilityBlock;
use shakmaty::{Board, Color, Move, Role, Square};

/// A block of features
#[enum_dispatch]
#[derive(Debug)]
pub enum FeatureBlocks {
    AxesBlock,
    PairwiseBlock,
    KingBlock,
    MobilityBlock,
}

#[enum_dispatch(FeatureBlocks)]
pub trait FeatureBlock {
    /// Size of the block
    fn size(&self) -> u16;

    /// Whether the given move requires a refresh of the features
    fn requires_refresh(
        &self,
        _board: &Board,
        _mov: &Move,
        _turn: Color,
        _perspective: Color,
    ) -> bool {
        false
    }

    /// Computes the initial features for the given board and perspective (potentially slow)
    fn active_features(
        &self,
        board: &Board,
        turn: Color,
        perspective: Color,
        features: &mut Vec<u16>,
        offset: u16,
    );

    /// Computes the features that have changed with the addition of the piece (hopefully fast)
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
    );

    /// Computes the features that have changed with the removal of the piece (hopefully fast)
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
    );
}
