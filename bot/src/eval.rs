use std::ops::Index;

use crate::{defs::Value, position::Position, search::Search};
use ndarray::Array2;
use ort::{inputs, Session};
use shakmaty::{Board, Chess, Color, Role};

pub struct NNAccumulator {}

pub struct NNModel {
    /// ML model to evaluate positions
    model: Session,
}

impl NNModel {
    pub fn from_memory(buffer: &[u8]) -> ort::Result<Self> {
        let session = Session::builder()?
            .with_intra_threads(1)?
            .with_model_from_memory(buffer)?;

        Ok(Self { model: session })
    }

    pub fn evaluate(&self, pos: &Position) -> Value {
        return basic_eval(pos);

        let mut board = pos.board().clone();
        if pos.turn() == Color::Black {
            board.flip_vertical();
            board.swap_colors();
        }

        let encoded = encode_board(&board);
        let mut x = Array2::<i64>::zeros((1, 12));
        let mut row = x.row_mut(0);

        for j in 0..12 {
            row[j] = encoded[j];
        }

        unsafe {
            let outputs = self
                .model
                .run(inputs![x].unwrap_unchecked())
                .unwrap_unchecked();
            let scores = outputs
                .index(0)
                .extract_raw_tensor::<f32>()
                .unwrap_unchecked()
                .1;
            let value = scores[0];

            return (value * 100.0) as Value;
        }
    }
}

fn encode_board(board: &Board) -> [i64; 12] {
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

fn basic_eval(position: &Position) -> Value {
    use shakmaty::Role::*;

    // pawn positional score
    const PAWN_SCORE: [i32; 64] = [
        90, 90, 90, 90, 90, 90, 90, 90, 30, 30, 30, 40, 40, 30, 30, 30, 20, 20, 20, 30, 30, 30, 20,
        20, 10, 10, 10, 20, 20, 10, 10, 10, 5, 5, 10, 20, 20, 5, 5, 5, 0, 0, 0, 5, 5, 0, 0, 0, 0,
        0, 0, -10, -10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ];
    const KNIGHT_SCORE: [i32; 64] = [
        -5, 0, 0, 0, 0, 0, 0, -5, -5, 0, 0, 10, 10, 0, 0, -5, -5, 5, 20, 20, 20, 20, 5, -5, -5, 10,
        20, 30, 30, 20, 10, -5, -5, 10, 20, 30, 30, 20, 10, -5, -5, 5, 20, 10, 10, 20, 5, -5, -5,
        0, 0, 0, 0, 0, 0, -5, -5, -10, 0, 0, 0, 0, -10, -5,
    ];
    const BISHOP_SCORE: [i32; 64] = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 10, 0, 0, 0, 0, 0, 10, 20, 20,
        10, 0, 0, 0, 0, 10, 20, 20, 10, 0, 0, 0, 10, 0, 0, 0, 0, 10, 0, 0, 30, 0, 0, 0, 0, 30, 0,
        0, 0, -10, 0, 0, -10, 0, 0,
    ];
    const ROOK_SCORE: [i32; 64] = [
        50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 0, 0, 10, 20, 20, 10, 0, 0,
        0, 0, 10, 20, 20, 10, 0, 0, 0, 0, 10, 20, 20, 10, 0, 0, 0, 0, 10, 20, 20, 10, 0, 0, 0, 0,
        10, 20, 20, 10, 0, 0, 0, 0, 0, 20, 20, 0, 0, 0,
    ];
    const KING_SCORE: [i32; 64] = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 0, 0, 0, 5, 5, 10, 10, 5, 5, 0, 0, 5, 10, 20, 20,
        10, 5, 0, 0, 5, 10, 20, 20, 10, 5, 0, 0, 0, 5, 10, 10, 5, 0, 0, 0, 5, 5, -5, -5, 0, 5, 0,
        0, 0, 5, 0, -15, 0, 10, 0,
    ];

    let mut score = 0;

    for (mut square, piece) in position.board().clone().into_iter() {
        let material_score = match piece.role {
            Pawn => 100,
            Knight => 300,
            Bishop => 350,
            Rook => 500,
            Queen => 1000,
            King => 10000,
        };

        if piece.color == Color::Black {
            square = square.flip_vertical();
        }

        let positional_score = match piece.role {
            Pawn => PAWN_SCORE[square as usize],
            Knight => KNIGHT_SCORE[square as usize],
            Bishop => BISHOP_SCORE[square as usize],
            Rook => ROOK_SCORE[square as usize],
            Queen => 0,
            King => KING_SCORE[square as usize],
        };

        let sign = if piece.color == position.turn() {
            1
        } else {
            -1
        };

        score += sign * (material_score + positional_score);
    }

    score
}
