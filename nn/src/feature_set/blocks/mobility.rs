use crate::feature_set::blocks::axes::correct_square;

use super::{king::KingBlock, FeatureBlock};
use shakmaty::{attacks, board, Board, Color, Move, MoveList, Piece, Role, Square};

#[derive(Debug)]
pub struct MobilityBlock {}

impl MobilityBlock {
    pub fn new() -> Self {
        Self {}
    }
}

impl FeatureBlock for MobilityBlock {
    fn size(&self) -> u16 {
        64 * 6 * 2
    }

    fn requires_refresh(
        &self,
        _board: &Board,
        _mov: &Move,
        _turn: Color,
        _perspective: Color,
    ) -> bool {
        true
    }

    fn active_features(
        &self,
        board: &Board,
        _turn: Color,
        perspective: Color,
        features: &mut Vec<u16>,
        offset: u16,
    ) {
        //let mut board = board.clone();

        //if perspective == Color::Black {
        //    // flip board
        //    board.flip_vertical();
        //    board.swap_colors();
        //}

        let occupied = board.occupied();
        let unoccupied = !occupied;

        for (piece_square, piece) in board.clone().into_iter() {
            let attack = attacks::attacks(piece_square, piece, occupied);
            let accesible = unoccupied.with(board.by_color(piece.color.other()));

            for to in attack & accesible {
                let to = correct_square(to, perspective);
                let role = piece.role as u16 - 1;
                let color = (piece.color != perspective) as u16;

                features.push(offset + to as u16 * 12 + role * 2 + color)
            }
        }
    }

    fn features_on_add(
        &self,
        board: &Board,
        mut piece_square: Square,
        piece_role: Role,
        mut piece_color: Color,
        perspective: Color,
        add_feats: &mut Vec<u16>,
        _rem_feats: &mut Vec<u16>,
        offset: u16,
    ) {
        panic!("Not implemented");
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
        panic!("Not implemented");
    }
}
