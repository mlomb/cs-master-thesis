use shakmaty::{Board, Color, Role};

pub fn encode_board(board: &Board) -> [i64; 12] {
    let mut data = [0 as i64; 12];

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

        data[channel] |= 1 << (square.file() as usize + (7 - square.rank() as usize) * 8);
    }

    data
}
