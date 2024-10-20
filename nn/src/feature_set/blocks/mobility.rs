use super::{all::correct_square, FeatureBlock};
use shakmaty::{attacks, Bitboard, Board, ByColor, ByRole, Color, Piece, Role, Square};

#[derive(Debug)]
pub struct MobilityBitsetBlock {}

impl MobilityBitsetBlock {
    pub fn new() -> Self {
        Self {}
    }

    pub fn compute_index(
        sq: Square,
        role: Role,
        color: Color,
        perspective: Color,
        offset: u16,
    ) -> u16 {
        let sq = correct_square(sq, perspective);
        let role = role as u16 - 1;
        let color = (color != perspective) as u16;

        offset + sq as u16 * 12 + role * 2 + color
    }

    #[inline(always)]
    pub fn update_features(
        prev_board: &Board,
        next_board: &Board,
        perspective: Color,
        add_feats: &mut Vec<u16>,
        rem_feats: &mut Vec<u16>,
        offset: u16,
    ) {
        let mobility_prev = mobility_by_role(prev_board);
        let mobility_next = mobility_by_role(next_board);

        for &role in Role::ALL.iter() {
            for &color in Color::ALL.iter() {
                let mobility_prev = *mobility_prev.get(role).get(color);
                let mobility_next = *mobility_next.get(role).get(color);

                for to in mobility_prev ^ mobility_next {
                    let index = Self::compute_index(to, role, color, perspective, offset);

                    if mobility_next.contains(to) {
                        add_feats.push(index);
                    } else if mobility_prev.contains(to) {
                        rem_feats.push(index);
                    }
                }
            }
        }
    }
}

impl FeatureBlock for MobilityBitsetBlock {
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
        let mobility = mobility_by_role(board);

        for &role in Role::ALL.iter() {
            for &color in Color::ALL.iter() {
                for sq in *mobility.get(role).get(color) {
                    let index = Self::compute_index(sq, role, color, perspective, offset);

                    features.push(index);
                }
            }
        }
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
        let mut next_board = board.clone();
        next_board.set_piece_at(
            piece_square,
            Piece {
                role: piece_role,
                color: piece_color,
            },
        );

        Self::update_features(
            &board,
            &next_board,
            perspective,
            add_feats,
            rem_feats,
            offset,
        );
    }

    fn features_on_remove(
        &self,
        board: &Board,
        piece_square: Square,
        _piece_role: Role,
        _piece_color: Color,
        perspective: Color,
        add_feats: &mut Vec<u16>,
        rem_feats: &mut Vec<u16>,
        offset: u16,
    ) {
        let mut next_board = board.clone();
        next_board.discard_piece_at(piece_square);

        Self::update_features(
            &board,
            &next_board,
            perspective,
            add_feats,
            rem_feats,
            offset,
        );
    }
}

const MOBILITY_COUNTS: [u16; 6] = [8, 15, 16, 25, 25, 8];
const MOBILITY_OFFSETS: [u16; 6] = [0, 9, 25, 42, 68, 94];

#[derive(Debug)]
pub struct MobilityCountsBlock {}

impl MobilityCountsBlock {
    pub fn new() -> Self {
        Self {}
    }

    pub fn compute_index(
        value: usize,
        role: Role,
        color: Color,
        perspective: Color,
        offset: u16,
    ) -> u16 {
        let role = role as u16 - 1;
        let color = (color != perspective) as u16;
        let bucket = (value as u16).min(MOBILITY_COUNTS[role as usize]);

        offset + MOBILITY_OFFSETS[role as usize] * 2 + bucket * 2 + color
    }

    #[inline(always)]
    pub fn update_features(
        prev_board: &Board,
        next_board: &Board,
        perspective: Color,
        add_feats: &mut Vec<u16>,
        rem_feats: &mut Vec<u16>,
        offset: u16,
    ) {
        let mobility_prev = mobility_by_role(prev_board);
        let mobility_next = mobility_by_role(next_board);

        for &role in Role::ALL.iter() {
            for &color in Color::ALL.iter() {
                let mobility_prev = (*mobility_prev.get(role).get(color)).count();
                let mobility_next = (*mobility_next.get(role).get(color)).count();

                if mobility_prev != mobility_next {
                    add_feats.push(Self::compute_index(
                        mobility_next,
                        role,
                        color,
                        perspective,
                        offset,
                    ));
                    rem_feats.push(Self::compute_index(
                        mobility_prev,
                        role,
                        color,
                        perspective,
                        offset,
                    ));
                }
            }
        }
    }
}

impl FeatureBlock for MobilityCountsBlock {
    fn size(&self) -> u16 {
        2 * (MOBILITY_COUNTS[0]
            + MOBILITY_COUNTS[1]
            + MOBILITY_COUNTS[2]
            + MOBILITY_COUNTS[3]
            + MOBILITY_COUNTS[4]
            + MOBILITY_COUNTS[5]
            + 6)
    }

    fn active_features(
        &self,
        board: &Board,
        _turn: Color,
        perspective: Color,
        features: &mut Vec<u16>,
        offset: u16,
    ) {
        let mobility = mobility_by_role(board);

        for &role in Role::ALL.iter() {
            for &color in Color::ALL.iter() {
                let count = (*mobility.get(role).get(color)).count();
                features.push(Self::compute_index(count, role, color, perspective, offset));
            }
        }
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
        let mut next_board = board.clone();
        next_board.set_piece_at(
            piece_square,
            Piece {
                role: piece_role,
                color: piece_color,
            },
        );

        Self::update_features(
            &board,
            &next_board,
            perspective,
            add_feats,
            rem_feats,
            offset,
        );
    }

    fn features_on_remove(
        &self,
        board: &Board,
        piece_square: Square,
        _piece_role: Role,
        _piece_color: Color,
        perspective: Color,
        add_feats: &mut Vec<u16>,
        rem_feats: &mut Vec<u16>,
        offset: u16,
    ) {
        let mut next_board = board.clone();
        next_board.discard_piece_at(piece_square);

        Self::update_features(
            &board,
            &next_board,
            perspective,
            add_feats,
            rem_feats,
            offset,
        );
    }
}

fn mobility_by_role(board: &Board) -> ByRole<ByColor<Bitboard>> {
    let mut all = ByRole::new_with(|_| ByColor::new_with(|_| Bitboard(0)));

    for (piece_square, piece) in board.clone().into_iter() {
        *all.get_mut(piece.role).get_mut(piece.color) |= mobility(&board, piece_square);
    }

    all
}

pub fn mobility(board: &Board, sq: Square) -> Bitboard {
    let piece = board.piece_at(sq).unwrap();
    let occupied = board.occupied();

    // all potential squares that can be attacked by the piece, regardless of color
    // . . P . Q . P .
    // . . 1 1 . 1 1 .
    // so we have to filter out the squares that are of another color
    let mut move_or_attack = attacks::attacks(sq, piece, occupied);

    // pawns are special
    // - they can only attack diagonally
    // - they can only move forward
    if piece.role == Role::Pawn {
        // only keep attacks to squares that contain an enemy piece
        move_or_attack &= board.by_color(piece.color.other());

        // add single step (which is not added by attacks::attacks)
        let single_step = sq.offset(piece.color.fold_wb(8, -8)).unwrap();

        // only if the square is not occupied
        if !occupied.contains(single_step) {
            move_or_attack.set(single_step, true);
        }
    }

    // filter out like said above
    let accesible = !board.by_color(piece.color);
    let mobility = move_or_attack & accesible;

    // filter out squares that would make the king go into check
    let mut safe_mobility = Bitboard(0);

    for to in mobility {
        let mut board = board.clone();
        let mut occupied = occupied.clone();

        board.discard_piece_at(sq);
        board.set_piece_at(to, piece);
        occupied.set(sq, false);
        occupied.set(to, true);

        // read the king square again, in case the piece we are moving is the king
        // so we prevent the king going into checks
        // NOTE: the king may not be in the board while inside of the FeatureSet::changed_features call
        if let Some(king_sq) = board.king_of(piece.color) {
            if board
                .attacks_to(king_sq, piece.color.other(), occupied)
                .is_empty()
            {
                safe_mobility.set(to, true);
            }
        }
    }

    // println!("move_or_attack post pawn\n{:?}", move_or_attack);
    // println!("accesible\n{:?}", accesible);
    // println!("safe_mobility\n{:?}", safe_mobility);

    safe_mobility
}

#[cfg(test)]
mod tests {
    use super::*;
    use shakmaty::{fen::Fen, Board, CastlingMode, Chess, Position};

    #[test]
    fn test_default() {
        let board = Board::default();

        // white pawns
        assert_eq!(mobility(&board, Square::A2).count(), 1);
        assert_eq!(mobility(&board, Square::B2).count(), 1);
        assert_eq!(mobility(&board, Square::C2).count(), 1);
        assert_eq!(mobility(&board, Square::D2).count(), 1);
        assert_eq!(mobility(&board, Square::E2).count(), 1);
        assert_eq!(mobility(&board, Square::F2).count(), 1);
        assert_eq!(mobility(&board, Square::G2).count(), 1);
        assert_eq!(mobility(&board, Square::H2).count(), 1);
        // white rest
        assert_eq!(mobility(&board, Square::A1).count(), 0);
        assert_eq!(mobility(&board, Square::B1).count(), 2);
        assert_eq!(mobility(&board, Square::C1).count(), 0);
        assert_eq!(mobility(&board, Square::D1).count(), 0);
        assert_eq!(mobility(&board, Square::E1).count(), 0);
        assert_eq!(mobility(&board, Square::F1).count(), 0);
        assert_eq!(mobility(&board, Square::G1).count(), 2);
        assert_eq!(mobility(&board, Square::H1).count(), 0);

        // black pawns
        assert_eq!(mobility(&board, Square::A7).count(), 1);
        assert_eq!(mobility(&board, Square::B7).count(), 1);
        assert_eq!(mobility(&board, Square::C7).count(), 1);
        assert_eq!(mobility(&board, Square::D7).count(), 1);
        assert_eq!(mobility(&board, Square::E7).count(), 1);
        assert_eq!(mobility(&board, Square::F7).count(), 1);
        assert_eq!(mobility(&board, Square::G7).count(), 1);
        assert_eq!(mobility(&board, Square::H7).count(), 1);
        // black rest
        assert_eq!(mobility(&board, Square::A8).count(), 0);
        assert_eq!(mobility(&board, Square::B8).count(), 2);
        assert_eq!(mobility(&board, Square::C8).count(), 0);
        assert_eq!(mobility(&board, Square::D8).count(), 0);
        assert_eq!(mobility(&board, Square::E8).count(), 0);
        assert_eq!(mobility(&board, Square::F8).count(), 0);
        assert_eq!(mobility(&board, Square::G8).count(), 2);
        assert_eq!(mobility(&board, Square::H8).count(), 0);
    }

    #[test]
    fn test_fen1() {
        let pos =
            Fen::from_ascii(b"r2b1rk1/3p1ppp/ppnqp1P1/2p5/2P1Q1P1/8/PP1PPP2/2RNBKR1 w - - 2 12")
                .unwrap()
                .into_position::<Chess>(CastlingMode::Standard)
                .unwrap();

        // white pieces
        assert_eq!(mobility(pos.board(), Square::C1).count(), 4);
        assert_eq!(mobility(pos.board(), Square::D1).count(), 2);
        assert_eq!(mobility(pos.board(), Square::E1).count(), 0);
        assert_eq!(mobility(pos.board(), Square::F1).count(), 1);
        assert_eq!(mobility(pos.board(), Square::G1).count(), 3);
        assert_eq!(mobility(pos.board(), Square::E4).count(), 14);
        // black pieces
        assert_eq!(mobility(pos.board(), Square::C6).count(), 7);
        assert_eq!(mobility(pos.board(), Square::D6).count(), 11);
        assert_eq!(mobility(pos.board(), Square::A8).count(), 3);
        assert_eq!(mobility(pos.board(), Square::D8).count(), 5);
        assert_eq!(mobility(pos.board(), Square::F8).count(), 1);
        assert_eq!(mobility(pos.board(), Square::G8).count(), 1);

        // white pawns
        assert_eq!(mobility(pos.board(), Square::A2).count(), 1);
        assert_eq!(mobility(pos.board(), Square::B2).count(), 1);
        assert_eq!(mobility(pos.board(), Square::D2).count(), 1);
        assert_eq!(mobility(pos.board(), Square::E2).count(), 1);
        assert_eq!(mobility(pos.board(), Square::F2).count(), 1);
        assert_eq!(mobility(pos.board(), Square::C4).count(), 0);
        assert_eq!(mobility(pos.board(), Square::G4).count(), 1);
        assert_eq!(mobility(pos.board(), Square::G6).count(), 2);
        // black pawns
        assert_eq!(mobility(pos.board(), Square::C5).count(), 0);
        assert_eq!(mobility(pos.board(), Square::A6).count(), 1);
        assert_eq!(mobility(pos.board(), Square::B6).count(), 1);
        assert_eq!(mobility(pos.board(), Square::C5).count(), 0);
        assert_eq!(mobility(pos.board(), Square::E6).count(), 1);
        assert_eq!(mobility(pos.board(), Square::D7).count(), 0);
        assert_eq!(mobility(pos.board(), Square::F7).count(), 2);
        assert_eq!(mobility(pos.board(), Square::G7).count(), 0);
        assert_eq!(mobility(pos.board(), Square::H7).count(), 2);
    }

    #[test]
    fn test_fen2() {
        let pos = Fen::from_ascii(b"6k1/1p3p2/pQq2bp1/3p3p/P3r3/2P1B1P1/1P3P2/R5K1 w - - 2 19")
            .unwrap()
            .into_position::<Chess>(CastlingMode::Standard)
            .unwrap();

        // white pieces
        assert_eq!(mobility(pos.board(), Square::A1).count(), 7);
        assert_eq!(mobility(pos.board(), Square::G1).count(), 4);
        assert_eq!(mobility(pos.board(), Square::E3).count(), 7);
        assert_eq!(mobility(pos.board(), Square::B6).count(), 12);
        // black pieces
        assert_eq!(mobility(pos.board(), Square::E4).count(), 12);
        assert_eq!(mobility(pos.board(), Square::C6).count(), 12);
        assert_eq!(mobility(pos.board(), Square::F6).count(), 9);
        assert_eq!(mobility(pos.board(), Square::G8).count(), 4);
    }

    #[test]
    fn test_fen3() {
        let pos =
            Fen::from_ascii(b"rnb1kbnr/pp1qpppp/3p4/1Bp5/4P3/5N1P/PPPP1PP1/RNBQK2R b KQkq - 0 4")
                .unwrap()
                .into_position::<Chess>(CastlingMode::Standard)
                .unwrap();

        // check that the queen has less mobility becasuse moving to some squares makes the king in check
        assert_eq!(mobility(pos.board(), Square::D7).count(), 2);
    }

    #[test]
    fn test_fen4() {
        let pos = Fen::from_ascii(b"6k1/4Q3/8/8/8/8/1K6/8 w - - 0 1")
            .unwrap()
            .into_position::<Chess>(CastlingMode::Standard)
            .unwrap();

        // check that the king can't move to squares that would make it go into check
        assert_eq!(mobility(pos.board(), Square::G8).count(), 1);
    }
}
