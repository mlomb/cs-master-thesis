use super::FeatureBlock;
use crate::feature_set::{axis::Axis, blocks::all::correct_square};
use shakmaty::{Board, Color, Piece, Role, Square};

/// A feature block where the features are the pairs of pieces on a given axis
/// (based on the order [not position] in the axis, the role and color)
///
/// Horizontal axis: vertical pairs
/// Vertical axis: horizontal pairs
#[derive(Debug)]
pub struct PairwiseBlock {
    axis: Axis,
}

impl PairwiseBlock {
    pub fn new(axis: Axis) -> Self {
        Self { axis }
    }

    #[inline(always)]
    fn find_pieces_on_axis(
        &self,
        board: &Board,
        mut piece_square: Square,
        mut piece: Piece,
        perspective: Color,
    ) -> (Option<Piece>, Piece, Option<Piece>) {
        let mut board = board.clone();

        if perspective == Color::Black {
            // flip board
            board.flip_vertical();
            board.swap_colors();
            piece_square = piece_square.flip_vertical();
            // flip piece color
            piece.color = piece.color.other();
        }

        let index = self.axis.index(piece_square);
        let bitboard = self.axis.bitboard(index);
        let pieces = board.occupied().intersect(bitboard);

        let mut left = None;
        let mut right = None;

        for sq in pieces {
            if sq < piece_square {
                left = Some(board.piece_at(sq).unwrap());
            } else if sq > piece_square {
                right = Some(board.piece_at(sq).unwrap());
                break;
            }
        }

        (left, piece, right)
    }

    #[inline(always)]
    fn compute_index(offset: u16, axis_index: u16, piece1: Piece, piece2: Piece) -> u16 {
        let piece1_role = piece1.role as u16 - 1;
        let piece1_color = piece1.color as u16;

        let piece2_role = piece2.role as u16 - 1;
        let piece2_color = piece2.color as u16;

        offset
            + axis_index * (6 * 2) * (6 * 2)
            + piece1_role * (2 * 6 * 2)
            + piece1_color * (6 * 2)
            + piece2_role * 2
            + piece2_color
    }
}

impl FeatureBlock for PairwiseBlock {
    fn size(&self) -> u16 {
        self.axis.size() * (6 * 2) * (6 * 2)
    }

    fn active_features(
        &self,
        board: &Board,
        _turn: Color,
        perspective: Color,
        features: &mut Vec<u16>,
        offset: u16,
    ) {
        let mut board = board.clone();
        if perspective == Color::Black {
            board.flip_vertical();
            board.swap_colors();
        }

        for index in 0..self.axis.size() {
            features.extend(
                board
                    .occupied()
                    .intersect(self.axis.bitboard(index))
                    .into_iter()
                    .map_windows(|[l, r]| {
                        Self::compute_index(
                            offset,
                            index,
                            board.piece_at(l.clone()).unwrap(),
                            board.piece_at(r.clone()).unwrap(),
                        )
                    }),
            );
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
        // We assume there is no piece at the square
        debug_assert!(board.piece_at(piece_square).is_none());

        let axis_index = self.axis.index(correct_square(piece_square, perspective));
        let triplet = self.find_pieces_on_axis(
            board,
            piece_square,
            Piece {
                role: piece_role,
                color: piece_color,
            },
            perspective,
        );

        // remove existing pair
        if let (Some(left), _, Some(right)) = triplet {
            rem_feats.push(Self::compute_index(offset, axis_index, left, right));
        }

        // add left pair
        if let (Some(left), piece, _) = triplet {
            add_feats.push(Self::compute_index(offset, axis_index, left, piece));
        }

        // add right pair
        if let (_, piece, Some(right)) = triplet {
            add_feats.push(Self::compute_index(offset, axis_index, piece, right));
        }
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
        // We assume there is a piece at the square
        debug_assert!(board.piece_at(piece_square).is_some());

        let axis_index = self.axis.index(correct_square(piece_square, perspective));
        let triplet = self.find_pieces_on_axis(
            board,
            piece_square,
            Piece {
                color: piece_color,
                role: piece_role,
            },
            perspective,
        );

        // remove pair with left piece
        if let (Some(left), piece, _) = triplet {
            rem_feats.push(Self::compute_index(offset, axis_index, left, piece));
        }

        // remove pair with right piece
        if let (_, piece, Some(right)) = triplet {
            rem_feats.push(Self::compute_index(offset, axis_index, piece, right));
        }

        // join left and right pieces
        if let (Some(left), _, Some(right)) = triplet {
            add_feats.push(Self::compute_index(offset, axis_index, left, right));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shakmaty::{fen::Fen, Board, CastlingMode, Chess, Position};

    fn num_pairs(board: &Board, perspective: Color, axis: Axis) -> usize {
        let block = PairwiseBlock::new(axis);
        let mut features = Vec::new();
        block.active_features(board, Color::White, perspective, &mut features, 0);
        features.len()
    }

    #[test]
    fn test_default() {
        let board = Board::default();

        assert_eq!(num_pairs(&board, Color::White, Axis::Horizontal), 3 * 8);
        assert_eq!(num_pairs(&board, Color::Black, Axis::Horizontal), 3 * 8);
        assert_eq!(num_pairs(&board, Color::White, Axis::Vertical), 4 * 7);
        assert_eq!(num_pairs(&board, Color::Black, Axis::Vertical), 4 * 7);
    }

    #[test]
    fn test_fen1() {
        let pos =
            Fen::from_ascii(b"r2b1rk1/3p1ppp/ppnqp1P1/2p5/2P1Q1P1/8/PP1PPP2/2RNBKR1 w - - 2 12")
                .unwrap()
                .into_position::<Chess>(CastlingMode::Standard)
                .unwrap();

        let vert = PairwiseBlock::new(Axis::Vertical);
        let horz = PairwiseBlock::new(Axis::Horizontal);

        let (l, _, r) = vert.find_pieces_on_axis(
            pos.board(),
            Square::C6,
            Piece {
                color: Color::Black,
                role: Role::Knight,
            },
            Color::White,
        );
        let (b, _, a) = horz.find_pieces_on_axis(
            pos.board(),
            Square::C6,
            Piece {
                color: Color::Black,
                role: Role::Knight,
            },
            Color::White,
        );

        // println!("L: {:?} R: {:?} A: {:?} B: {:?}", l, r, a, b);

        assert_eq!(
            l,
            Some(Piece {
                color: Color::Black,
                role: Role::Pawn
            })
        );
        assert_eq!(
            r,
            Some(Piece {
                color: Color::Black,
                role: Role::Queen
            })
        );
        assert_eq!(a, None);
        assert_eq!(
            b,
            Some(Piece {
                color: Color::Black,
                role: Role::Pawn
            })
        );

        assert_eq!(num_pairs(&pos.board(), Color::White, Axis::Horizontal), 21);
        assert_eq!(num_pairs(&pos.board(), Color::Black, Axis::Horizontal), 21);
        assert_eq!(num_pairs(&pos.board(), Color::White, Axis::Vertical), 20);
        assert_eq!(num_pairs(&pos.board(), Color::Black, Axis::Vertical), 20);
    }

    #[test]
    fn test_fen2() {
        let pos = Fen::from_ascii(b"6k1/1p3p2/pQq2bp1/3p3p/P3r3/2P1B1P1/1P3P2/R5K1 w - - 2 19")
            .unwrap()
            .into_position::<Chess>(CastlingMode::Standard)
            .unwrap();

        let vert = PairwiseBlock::new(Axis::Vertical);
        let horz = PairwiseBlock::new(Axis::Horizontal);

        let (l, _, r) = vert.find_pieces_on_axis(
            pos.board(),
            Square::D5,
            Piece {
                color: Color::Black,
                role: Role::Pawn,
            },
            Color::White,
        );
        let (b, _, a) = horz.find_pieces_on_axis(
            pos.board(),
            Square::D5,
            Piece {
                color: Color::Black,
                role: Role::Pawn,
            },
            Color::White,
        );
        assert_eq!(l, None);
        assert_eq!(
            r,
            Some(Piece {
                color: Color::Black,
                role: Role::Pawn
            })
        );
        assert_eq!(a, None);
        assert_eq!(b, None);

        assert_eq!(num_pairs(&pos.board(), Color::White, Axis::Horizontal), 11);
        assert_eq!(num_pairs(&pos.board(), Color::Black, Axis::Horizontal), 11);
        assert_eq!(num_pairs(&pos.board(), Color::White, Axis::Vertical), 11);
        assert_eq!(num_pairs(&pos.board(), Color::Black, Axis::Vertical), 11);
    }
}
