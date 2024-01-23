use shakmaty::{Board, Color, Role};
use std::{
    fs::File,
    io::{self, Write},
};

#[derive(Debug)]
pub struct Sample {
    pub parent: Board,
    pub observed: Board,
    pub random: Board,
}

fn encode_board(board: &Board) -> [u8; 8 * 12] {
    let mut data = [0 as u8; 8 * 12];

    for (square, piece) in board.clone().into_iter() {
        let channel = match piece.color {
            Color::White => match piece.role {
                Role::Pawn => 0,
                Role::Knight => 1,
                Role::Bishop => 2,
                Role::Rook => 3,
                Role::Queen => 4,
                Role::King => 5,
            },
            Color::Black => match piece.role {
                Role::Pawn => 6,
                Role::Knight => 7,
                Role::Bishop => 8,
                Role::Rook => 9,
                Role::Queen => 10,
                Role::King => 11,
            },
        };

        data[7 - square.rank() as usize + channel * 8] |= 1 << square.file() as usize;
    }

    data
}

impl Sample {
    pub fn write_to(&self, file: &mut File) -> io::Result<()> {
        file.write_all(&encode_board(&self.parent))?;
        file.write_all(&encode_board(&self.observed))?;
        file.write_all(&encode_board(&self.random))?;

        Ok(())
    }
}
