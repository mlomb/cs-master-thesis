mod axis;
mod blocks;
pub mod build;
mod checks;

use blocks::{FeatureBlock, FeatureBlocks};
use shakmaty::{Board, Color, File, Move, Piece, Role, Square};

/// A set of features for a neural network
#[derive(Debug)]
pub struct FeatureSet {
    /// Blocks of features that are added/concatenated together
    blocks: Vec<FeatureBlocks>,
}

impl FeatureSet {
    /// Create a feature set from the sum of feature blocks
    pub fn sum_of(blocks: Vec<FeatureBlocks>) -> Self {
        Self { blocks }
    }

    /// Number of features in the set
    #[inline(always)]
    pub fn num_features(&self) -> u16 {
        self.blocks.iter().map(|b| b.size()).sum::<u16>()
    }

    /// Whether the given move requires a refresh of the features
    #[inline(always)]
    pub fn requires_refresh(
        &self,
        board: &Board,
        mov: &Move,
        turn: Color,
        perspective: Color,
    ) -> bool {
        self.blocks
            .iter()
            .any(|b| b.requires_refresh(board, mov, turn, perspective))
    }

    /// Computes the initial features for the given board and perspective (potentially slow)
    #[inline(always)]
    pub fn active_features(
        &self,
        board: &Board,
        turn: Color,
        perspective: Color,
        features: &mut Vec<u16>,
    ) {
        let mut offset = 0;

        for block in &self.blocks {
            block.active_features(board, turn, perspective, features, offset);
            offset += block.size();
        }
    }

    /// Computes the features that have changed with the given move (hopefully fast)
    #[inline(always)]
    pub fn changed_features(
        &self,
        board: &Board,
        mov: &Move,
        turn: Color,
        perspective: Color,
        add_feats: &mut Vec<u16>,
        rem_feats: &mut Vec<u16>,
    ) {
        let mut board = board.clone();
        let from = mov.from().unwrap();
        let to = mov.to();
        let who_plays = turn;

        if mov.is_en_passant() {
            self.remove_piece(
                &mut board,
                Square::from_coords(to.file(), from.rank()),
                Role::Pawn,
                who_plays.other(),
                perspective,
                add_feats,
                rem_feats,
            );
        } else if let Some(captured) = mov.capture() {
            self.remove_piece(
                &mut board,
                to,
                captured,
                who_plays.other(),
                perspective,
                add_feats,
                rem_feats,
            );
        }

        match mov {
            Move::Normal { from, to, .. } | Move::EnPassant { from, to } => {
                let final_role = mov.promotion().unwrap_or(mov.role());

                self.add_piece(
                    &mut board,
                    *to,
                    final_role,
                    who_plays,
                    perspective,
                    add_feats,
                    rem_feats,
                );
                self.remove_piece(
                    &mut board,
                    *from,
                    mov.role(),
                    who_plays,
                    perspective,
                    add_feats,
                    rem_feats,
                )
            }
            Move::Castle { king, rook } => {
                self.remove_piece(
                    &mut board,
                    *king,
                    Role::King,
                    who_plays,
                    perspective,
                    add_feats,
                    rem_feats,
                );
                self.remove_piece(
                    &mut board,
                    *rook,
                    Role::Rook,
                    who_plays,
                    perspective,
                    add_feats,
                    rem_feats,
                );

                let (king_file, rook_file) = if king < rook {
                    // king side
                    (File::G, File::F)
                } else {
                    // queen side
                    (File::C, File::D)
                };

                self.add_piece(
                    &mut board,
                    Square::from_coords(king_file, king.rank()),
                    Role::King,
                    who_plays,
                    perspective,
                    add_feats,
                    rem_feats,
                );
                self.add_piece(
                    &mut board,
                    Square::from_coords(rook_file, rook.rank()),
                    Role::Rook,
                    who_plays,
                    perspective,
                    add_feats,
                    rem_feats,
                );
            }
            _ => unreachable!(),
        }

        // hacky optimization: remove common features
        // while rem_feats.last().is_some() && add_feats.last() == rem_feats.last() {
        //     add_feats.pop();
        //     rem_feats.pop();
        // }
    }

    /// Add a piece to the board and update the features
    #[inline(always)]
    fn add_piece(
        &self,
        board: &mut Board,
        piece_square: Square,
        piece_role: Role,
        piece_color: Color,
        perspective: Color,
        add_feats: &mut Vec<u16>,
        rem_feats: &mut Vec<u16>,
    ) {
        let mut offset = 0;

        for block in &self.blocks {
            block.features_on_add(
                board,
                piece_square,
                piece_role,
                piece_color,
                perspective,
                add_feats,
                rem_feats,
                offset,
            );
            offset += block.size();
        }

        board.set_piece_at(
            piece_square,
            Piece {
                role: piece_role,
                color: piece_color,
            },
        );
    }

    /// Remove a piece from the board and update the features
    #[inline(always)]
    fn remove_piece(
        &self,
        board: &mut Board,
        piece_square: Square,
        piece_role: Role,
        piece_color: Color,
        perspective: Color,
        add_feats: &mut Vec<u16>,
        rem_feats: &mut Vec<u16>,
    ) {
        let mut offset = 0;

        for block in &self.blocks {
            block.features_on_remove(
                board,
                piece_square,
                piece_role,
                piece_color,
                perspective,
                add_feats,
                rem_feats,
                offset,
            );
            offset += block.size();
        }

        board.discard_piece_at(piece_square);
    }
}
