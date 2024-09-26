use super::{all::correct_square, FeatureBlock};
use shakmaty::{attacks, Board, Color, Role, Square};

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

    fn active_features(
        &self,
        board: &Board,
        _turn: Color,
        perspective: Color,
        features: &mut Vec<u16>,
        offset: u16,
    ) {
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
        piece_square: Square,
        _piece_role: Role,
        piece_color: Color,
        perspective: Color,
        add_feats: &mut Vec<u16>,
        rem_feats: &mut Vec<u16>,
        offset: u16,
    ) {
        let occupied_prev = board.occupied();
        let mut occupied_next = board.occupied();
        occupied_next.set(piece_square, true);

        let unoccupied_prev = !occupied_prev;
        let unoccupied_next = !occupied_next;

        let accesible_prev = [
            unoccupied_prev.with(board.by_color(Color::White)),
            unoccupied_prev.with(board.by_color(Color::Black)),
        ];
        let mut accesible_next = [
            unoccupied_next.with(board.by_color(Color::White)),
            unoccupied_next.with(board.by_color(Color::Black)),
        ];
        accesible_next[piece_color as usize].set(piece_square, true);

        for (piece_square, piece) in board.clone().into_iter() {
            let attack_prev = attacks::attacks(piece_square, piece, occupied_prev);
            let attack_next = attacks::attacks(piece_square, piece, occupied_next);

            let mobility_prev = attack_prev & accesible_prev[piece.color.other() as usize];
            let mobility_next = attack_next & accesible_next[piece.color.other() as usize];

            for to in mobility_prev ^ mobility_next {
                let to = correct_square(to, perspective);
                let role = piece.role as u16 - 1;
                let color = (piece.color != perspective) as u16;

                let index = offset + to as u16 * 12 + role * 2 + color;

                if mobility_next.contains(to) {
                    add_feats.push(index);
                } else if mobility_prev.contains(to) {
                    rem_feats.push(index);
                }
            }
        }
    }

    fn features_on_remove(
        &self,
        board: &Board,
        piece_square: Square,
        _piece_role: Role,
        piece_color: Color,
        perspective: Color,
        add_feats: &mut Vec<u16>,
        rem_feats: &mut Vec<u16>,
        offset: u16,
    ) {
        let occupied_prev = board.occupied();
        let mut occupied_next = board.occupied();
        occupied_next.set(piece_square, false);

        let unoccupied_prev = !occupied_prev;
        let unoccupied_next = !occupied_next;

        let accesible_prev = [
            unoccupied_prev.with(board.by_color(Color::White)),
            unoccupied_prev.with(board.by_color(Color::Black)),
        ];
        let mut accesible_next = [
            unoccupied_next.with(board.by_color(Color::White)),
            unoccupied_next.with(board.by_color(Color::Black)),
        ];
        accesible_next[piece_color as usize].set(piece_square, false);

        for (piece_square, piece) in board.clone().into_iter() {
            let attack_prev = attacks::attacks(piece_square, piece, occupied_prev);
            let attack_next = attacks::attacks(piece_square, piece, occupied_next);

            let mobility_prev = attack_prev & accesible_prev[piece.color.other() as usize];
            let mobility_next = attack_next & accesible_next[piece.color.other() as usize];

            for to in mobility_prev ^ mobility_next {
                let to = correct_square(to, perspective);
                let role = piece.role as u16 - 1;
                let color = (piece.color != perspective) as u16;

                let index = offset + to as u16 * 12 + role * 2 + color;

                if mobility_next.contains(to) {
                    add_feats.push(index);
                } else if mobility_prev.contains(to) {
                    rem_feats.push(index);
                }
            }
        }
    }
}
