use shakmaty::{Bitboard, File, Rank, Square};

#[derive(Debug)]
pub enum Axis {
    /// Across files (↔)
    Horizontal,
    /// Across ranks (↕)
    Vertical,
    /// Forward diagonal (/)
    Diagonal1,
    /// Backward diagonal (\)
    Diagonal2,
}

impl Axis {
    /// Size of the axis dimension
    #[inline(always)]
    pub const fn size(&self) -> u16 {
        match self {
            Self::Horizontal => 8,
            Self::Vertical => 8,
            Self::Diagonal1 => 15,
            Self::Diagonal2 => 15,
        }
    }

    /// Index of the square based on the axis
    #[inline(always)]
    pub fn index(&self, piece_square: Square) -> u16 {
        let file = piece_square.file() as u16;
        let rank = piece_square.rank() as u16;

        match self {
            Self::Horizontal => file,
            Self::Vertical => rank,
            Self::Diagonal1 => file + rank,
            Self::Diagonal2 => file + 7 - rank,
        }
    }

    /// Bitboard of the axis
    pub fn bitboard(&self, index: u16) -> Bitboard {
        match self {
            Self::Horizontal => Bitboard::from_file(File::new(index as u32)),
            Self::Vertical => Bitboard::from_rank(Rank::new(index as u32)),
            _ => todo!(),
        }
    }
}
