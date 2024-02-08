use crate::{defs::Value, position::Position};
use ort::{inputs, Session};
use shakmaty::{Board, Color, Role};
use std::{ops::Index, usize};

// BLAS
use cblas_sys;
use cblas_sys::{CblasRowMajor, CblasTrans};

pub struct NNAccumulator {}

enum Activation {
    None,
    ReLU,
}

struct DenseLayer<const R: usize, const C: usize> {
    weights: Vec<f32>,
    bias: Vec<f32>,
    activation: Activation,
}

impl<const R: usize, const C: usize> DenseLayer<R, C> {
    fn new(weights: Vec<f32>, bias: Vec<f32>, activation: Activation) -> Self {
        assert!(weights.len() == R * C);
        assert!(bias.len() == C);
        Self {
            weights,
            bias,
            activation,
        }
    }

    fn forward(&self, input: &[f32], output: &mut [f32]) {
        debug_assert!(input.len() == R);
        debug_assert!(output.len() == C);

        unsafe {
            cblas_sys::cblas_sgemv(
                CblasRowMajor,
                CblasTrans,
                R as i32,
                C as i32,
                1.0,
                self.weights.as_ptr(),
                C as i32,
                input.as_ptr(),
                1,
                0.0,
                output.as_mut_ptr(),
                1,
            );
        }

        // use cols because weights are transposed

        match self.activation {
            Activation::None => {
                for i in 0..C {
                    let o = unsafe { output.get_unchecked_mut(i) };
                    let b = unsafe { self.bias.get_unchecked(i) };
                    *o += *b;
                }
            }
            Activation::ReLU => {
                for i in 0..C {
                    let o = unsafe { output.get_unchecked_mut(i) };
                    let b = unsafe { self.bias.get_unchecked(i) };
                    *o = (*o + *b).max(0.0);
                }
            }
        }
    }
}

fn to_f32_vec(bytes: &[u8]) -> Vec<f32> {
    assert!(bytes.len() % 4 == 0);

    bytes
        .chunks(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect::<Vec<f32>>()
}

pub struct NNModel {
    /// ML model to evaluate positions
    model: Session,

    dense1: DenseLayer<768, 256>,
    dense2: DenseLayer<256, 256>,
    dense3: DenseLayer<256, 256>,
    dense4: DenseLayer<256, 1>,

    x: [f32; 768],
    h1: [f32; 256],
    h2: [f32; 256],
    h3: [f32; 256],
    h4: [f32; 1],
}

impl NNModel {
    pub fn from_memory(buffer: &[u8]) -> ort::Result<Self> {
        let session = Session::builder()?
            .with_intra_threads(1)?
            .with_model_from_memory(buffer)?;

        Ok(Self {
            model: session,

            dense1: DenseLayer::new(
                to_f32_vec(include_bytes!(
                    "../../models/rq-mse-256-0.470/dense_kernel_768x256.bin"
                )),
                to_f32_vec(include_bytes!(
                    "../../models/rq-mse-256-0.470/dense_bias_256.bin"
                )),
                Activation::ReLU,
            ),
            dense2: DenseLayer::new(
                to_f32_vec(include_bytes!(
                    "../../models/rq-mse-256-0.470/dense_1_kernel_256x256.bin"
                )),
                to_f32_vec(include_bytes!(
                    "../../models/rq-mse-256-0.470/dense_1_bias_256.bin"
                )),
                Activation::ReLU,
            ),
            dense3: DenseLayer::new(
                to_f32_vec(include_bytes!(
                    "../../models/rq-mse-256-0.470/dense_2_kernel_256x256.bin"
                )),
                to_f32_vec(include_bytes!(
                    "../../models/rq-mse-256-0.470/dense_2_bias_256.bin"
                )),
                Activation::ReLU,
            ),
            dense4: DenseLayer::new(
                to_f32_vec(include_bytes!(
                    "../../models/rq-mse-256-0.470/dense_3_kernel_256x1.bin"
                )),
                to_f32_vec(include_bytes!(
                    "../../models/rq-mse-256-0.470/dense_3_bias_1.bin"
                )),
                Activation::None,
            ),

            x: [0.0; 768],
            h1: [0.0; 256],
            h2: [0.0; 256],
            h3: [0.0; 256],
            h4: [0.0; 1],
        })
    }

    pub fn evaluate(&mut self, pos: &Position) -> Value {
        return (self.new_eval(pos) * 100.0) as Value;

        println!("ref: {} new: {}", self.ref_eval(pos), self.new_eval(pos));

        {
            let start = std::time::Instant::now();
            for _ in 0..1000 {
                self.new_eval(pos);
            }
            println!("new: {:?}", start.elapsed());
        }
        {
            let start = std::time::Instant::now();
            for _ in 0..1000 {
                self.ref_eval(pos);
            }
            println!("ref: {:?}", start.elapsed());
        }

        return basic_eval(pos);
    }

    fn new_eval(&mut self, pos: &Position) -> f32 {
        let mut board = pos.board().clone();
        if pos.turn() == Color::Black {
            board.flip_vertical();
            board.swap_colors();
        }

        self.x.fill(0.0);

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

            self.x[channel * 64 + (square.file() as usize + (7 - square.rank() as usize) * 8)] =
                1.0;
        }

        self.dense1.forward(&self.x, &mut self.h1);
        self.dense2.forward(&self.h1, &mut self.h2);
        self.dense3.forward(&self.h2, &mut self.h3);
        self.dense4.forward(&self.h3, &mut self.h4);

        self.h4[0]
    }

    fn ref_eval(&self, pos: &Position) -> f32 {
        use ndarray::Array2;

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

            return value;
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
