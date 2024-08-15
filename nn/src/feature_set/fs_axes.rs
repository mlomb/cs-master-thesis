use super::FeatureSet;
use shakmaty::{Board, Color, Square};

pub enum Axes {
    // Across files (↔)
    Horizontal,
    // Across ranks (↕)
    Vertical,
    // Forward diagonal (/)
    Diagonal1,
    // Backward diagonal (\)
    Diagonal2,
}

impl Axes {
    // Number of indexable steps
    pub fn size(&self) -> usize {
        match self {
            Axes::Horizontal => 8,
            Axes::Vertical => 8,
            Axes::Diagonal1 => 15,
            Axes::Diagonal2 => 15,
        }
    }

    pub fn index(&self, piece_square: Square) -> u16 {
        let file = piece_square.file() as u16;
        let rank = piece_square.rank() as u16;

        match self {
            Axes::Horizontal => file,
            Axes::Vertical => rank,
            Axes::Diagonal1 => file + rank,
            Axes::Diagonal2 => file + 7 - rank,
        }
    }
}

pub struct AxesBlock {
    pub first_axis: Axes,
    pub second_axis: Option<Axes>,
}

impl AxesBlock {
    pub fn size(&self) -> usize {
        self.first_axis.size() * self.second_axis.as_ref().map_or(1, |ax| ax.size()) * 6 * 2
    }
}

pub struct AxesFeatureSet {
    pub blocks: Vec<AxesBlock>,
}

impl FeatureSet for AxesFeatureSet {
    fn num_features(&self) -> usize {
        self.blocks.iter().map(|b| b.size()).sum::<usize>()
    }

    fn active_features(&self, board: &Board, perspective: Color, features: &mut Vec<u16>) {
        for (piece_square, piece) in board.clone().into_iter() {
            let piece_square = if perspective == Color::Black {
                // flip square vertically if black is to play, so it is on the bottom side
                piece_square.flip_vertical()
            } else {
                // keep square as is, by default white is below
                piece_square
            };

            let piece_file = piece_square.file() as u16;
            let piece_rank = piece_square.rank() as u16;
            let piece_diag1 = piece_file + piece_rank;
            let piece_diag2 = piece_file + 7 - piece_rank;

            debug_assert!(piece_diag1 < 15);
            debug_assert!(piece_diag2 < 15);

            let piece_role = piece.role as u16 - 1;
            let piece_color = (piece.color != perspective) as u16;

            let mut offset: usize = 0;

            for block in &self.blocks {
                let first_index = block.first_axis.index(piece_square);

                let index = match block.second_axis {
                    None => first_index,
                    Some(ref second_axis) => {
                        first_index * second_axis.size() as u16 + second_axis.index(piece_square)
                    }
                };

                features.push(offset as u16 + index * 12 + piece_role * 2 + piece_color);
                offset += block.size();
            }
        }
    }
}
