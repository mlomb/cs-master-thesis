use super::FeatureSet;
use shakmaty::{Board, Color, File, Move, Role, Square};

#[derive(PartialEq, Eq)]
pub enum Axes {
    // Across files (↔)
    Horizontal,
    // Across ranks (↕)
    Vertical,
    // Forward diagonal (/)
    Diagonal1,
    // Backward diagonal (\)
    Diagonal2,
    // King
    King,
}

#[inline(always)]
fn correct_square(piece_square: Square, perspective: Color) -> Square {
    if perspective == Color::Black {
        // flip square vertically if black is to play, so it is on the bottom side
        piece_square.flip_vertical()
    } else {
        // keep square as is, by default white is below
        piece_square
    }
}

impl Axes {
    // Number of indexable steps
    #[inline(always)]
    pub fn size(&self) -> usize {
        match self {
            Axes::Horizontal => 8,
            Axes::Vertical => 8,
            Axes::Diagonal1 => 15,
            Axes::Diagonal2 => 15,
            Axes::King => 64,
        }
    }

    #[inline(always)]
    pub fn index(&self, board: &Board, perspective: Color, piece_square: Square) -> u16 {
        let file = piece_square.file() as u16;
        let rank = piece_square.rank() as u16;

        match self {
            Axes::Horizontal => file,
            Axes::Vertical => rank,
            Axes::Diagonal1 => file + rank,
            Axes::Diagonal2 => file + 7 - rank,
            Axes::King => correct_square(board.king_of(perspective).unwrap(), perspective) as u16,
        }
    }
}

pub struct AxesBlock {
    pub axes: Vec<Axes>,
    pub incl_king: bool,
}

impl AxesBlock {
    #[inline(always)]
    pub fn size(&self) -> usize {
        self.axes.iter().map(|ax| ax.size()).product::<usize>()
            * (if self.incl_king { 6 } else { 5 })
            * 2
    }
}

pub struct AxesFeatureSet {
    pub blocks: Vec<AxesBlock>,
}

impl AxesFeatureSet {
    #[inline(always)]
    fn compute_indexes(
        &self,
        board: &Board,
        piece_square: Square,
        piece_role_: Role,
        piece_color: Color,
        perspective: Color,
        features: &mut Vec<u16>,
    ) {
        let piece_square = correct_square(piece_square, perspective);
        let piece_file = piece_square.file() as u16;
        let piece_rank = piece_square.rank() as u16;
        let piece_diag1 = piece_file + piece_rank;
        let piece_diag2 = piece_file + 7 - piece_rank;

        debug_assert!(piece_diag1 < 15);
        debug_assert!(piece_diag2 < 15);

        let piece_role = piece_role_ as u16 - 1;
        let piece_color = (piece_color != perspective) as u16;

        // block offset
        let mut offset: usize = 0;

        for block in &self.blocks {
            if !block.incl_king && piece_role_ == Role::King {
                // skip king
                continue;
            }

            let mut index: u16 = 0;

            for ax in &block.axes {
                index = index * ax.size() as u16 + ax.index(board, perspective, piece_square);
            }

            features.push(
                offset as u16
                    + index * (if block.incl_king { 12 } else { 10 })
                    + piece_role * 2
                    + piece_color,
            );
            offset += block.size();
        }
    }
}

impl FeatureSet for AxesFeatureSet {
    fn num_features(&self) -> usize {
        self.blocks.iter().map(|b| b.size()).sum::<usize>()
    }

    fn requires_refresh(
        &self,
        _board: &Board,
        mov: &Move,
        turn: Color,
        perspective: Color,
    ) -> bool {
        if mov.role() == Role::King && turn == perspective {
            for block in &self.blocks {
                for ax in &block.axes {
                    if *ax == Axes::King {
                        return true;
                    }
                }
            }
        }

        false
    }

    #[inline(always)]
    fn active_features(
        &self,
        board: &Board,
        _turn: Color,
        perspective: Color,
        features: &mut Vec<u16>,
    ) {
        for (piece_square, piece) in board.clone().into_iter() {
            self.compute_indexes(
                board,
                piece_square,
                piece.role,
                piece.color,
                perspective,
                features,
            );
        }
    }

    #[inline(always)]
    fn changed_features(
        &self,
        board: &Board,
        mov: &Move,
        turn: Color,
        perspective: Color,
        add_feats: &mut Vec<u16>,
        rem_feats: &mut Vec<u16>,
    ) {
        let from = mov.from().unwrap();
        let to = mov.to();
        let who_plays = turn;

        match mov {
            Move::Normal { from, to, .. } | Move::EnPassant { from, to } => {
                let final_role = mov.promotion().unwrap_or(mov.role());

                self.compute_indexes(&board, *to, final_role, who_plays, perspective, add_feats);
                self.compute_indexes(&board, *from, mov.role(), who_plays, perspective, rem_feats);
            }
            Move::Castle { king, rook } => {
                self.compute_indexes(&board, *king, Role::King, who_plays, perspective, rem_feats);
                self.compute_indexes(&board, *rook, Role::Rook, who_plays, perspective, rem_feats);

                let (king_file, rook_file) = if king < rook {
                    // king side
                    (File::G, File::F)
                } else {
                    // queen side
                    (File::C, File::D)
                };

                self.compute_indexes(
                    &board,
                    Square::from_coords(king_file, king.rank()),
                    Role::King,
                    who_plays,
                    perspective,
                    add_feats,
                );
                self.compute_indexes(
                    &board,
                    Square::from_coords(rook_file, rook.rank()),
                    Role::Rook,
                    who_plays,
                    perspective,
                    add_feats,
                );
                return;
            }
            _ => unreachable!(),
        }

        if mov.is_en_passant() {
            self.compute_indexes(
                &board,
                Square::from_coords(to.file(), from.rank()),
                Role::Pawn,
                who_plays.other(),
                perspective,
                rem_feats,
            );
        } else if let Some(captured) = mov.capture() {
            self.compute_indexes(
                &board,
                to,
                captured,
                who_plays.other(),
                perspective,
                rem_feats,
            );
        }
    }
}
