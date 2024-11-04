use super::{Sample, SampleEncoder};
use crate::pos_encoding::{encode_position, encoded_size};
use nn::feature_set::FeatureSet;
use rand::seq::SliceRandom;
use shakmaty::{Color, Position};
use std::io::Write;

const P: u32 = 0; // 0, 25, 50, 75

pub struct PQREncoding;

impl SampleEncoder for PQREncoding {
    fn x_size(&self, feature_set: &FeatureSet) -> usize {
        3 * encoded_size(feature_set)
    }

    fn y_size(&self) -> usize {
        0
    }

    fn write_sample(
        &self,
        sample: &Sample,
        write_x: &mut dyn Write,
        _write_y: &mut dyn Write,
        feature_set: &FeatureSet,
    ) {
        let mut rng = rand::thread_rng();

        // P: initial
        let p_position = &sample.position;
        let moves = p_position.legal_moves();

        let m = get_m(
            sample.position.turn(),
            u32::from(sample.position.fullmoves()),
        );

        if (moves.len() as u32) < m {
            // filter out or not enough moves to choose from
            return;
        }

        // Q: best
        let q_move = &sample.bestmove;
        let q_position = p_position.clone().play(q_move).unwrap();

        // R: random, different from Q
        let r_position = loop {
            // find a random R position
            let r_move = moves.choose(&mut rng).unwrap();
            if r_move == q_move {
                // R != Q
                continue;
            }

            break p_position.clone().play(&r_move).unwrap();
        };

        assert_eq!(p_position.turn().other(), q_position.turn());
        assert_eq!(p_position.turn().other(), r_position.turn());

        encode_position(&p_position, feature_set, write_x);
        encode_position(&q_position, feature_set, write_x);
        encode_position(&r_position, feature_set, write_x);
    }
}

fn get_m(color: Color, fullmoves: u32) -> u32 {
    let v = match color {
        Color::White => match P {
            0 => vec![
                0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            ],
            25 => vec![
                0, 0, 0, 0, 0, 0, 0, 0, 0, 34, 34, 34, 34, 34, 34, 34, 33, 33, 32, 32, 31, 30, 29,
                28, 27, 27, 26, 25, 24, 23, 22, 21, 21, 20, 19, 18, 18, 17, 17, 16, 16, 16, 15, 15,
                15, 14, 14, 14, 13, 13, 13, 13, 13, 13, 12, 12, 12, 12, 12, 12, 12, 11, 11, 11, 11,
                11, 11, 11, 11, 11, 11, 11, 11, 10, 11, 11, 11, 11, 10, 10, 10, 10, 10, 10, 10, 10,
                10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
            ],
            50 => vec![
                0, 0, 0, 0, 0, 0, 0, 0, 0, 39, 39, 39, 39, 39, 39, 38, 38, 38, 37, 37, 36, 36, 35,
                35, 34, 33, 32, 31, 31, 30, 29, 28, 27, 27, 26, 25, 25, 24, 23, 23, 22, 22, 21, 21,
                21, 20, 20, 20, 20, 19, 19, 19, 19, 19, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 17,
                17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17,
                16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
            ],
            75 => vec![
                0, 0, 0, 0, 0, 0, 0, 0, 0, 43, 43, 43, 43, 43, 43, 43, 43, 43, 42, 42, 42, 41, 41,
                40, 40, 39, 38, 38, 37, 36, 36, 35, 34, 33, 33, 32, 31, 31, 30, 29, 29, 28, 28, 27,
                27, 27, 26, 26, 26, 25, 25, 25, 25, 25, 24, 24, 24, 24, 24, 24, 24, 23, 23, 23, 23,
                23, 23, 23, 23, 23, 23, 23, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22,
                22, 22, 22, 22, 22, 22, 21, 21, 21, 21, 21, 21, 21, 21, 21,
            ],
            _ => panic!("Invalid p value"),
        },
        Color::Black => match P {
            0 => vec![
                0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            ],
            25 => vec![
                0, 0, 0, 0, 0, 0, 0, 0, 0, 31, 31, 31, 31, 31, 31, 30, 30, 30, 29, 29, 28, 27, 27,
                26, 25, 24, 23, 22, 21, 20, 19, 19, 18, 17, 17, 16, 15, 15, 14, 14, 14, 13, 13, 12,
                12, 12, 11, 11, 11, 11, 11, 10, 10, 10, 10, 10, 10, 10, 9, 9, 9, 9, 9, 9, 9, 9, 9,
                9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
                9, 9, 9, 9, 9, 9,
            ],
            50 => vec![
                0, 0, 0, 0, 0, 0, 0, 0, 0, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 34, 34, 33, 33,
                32, 31, 31, 30, 29, 28, 27, 27, 26, 25, 24, 24, 23, 22, 22, 21, 21, 20, 20, 19, 19,
                19, 18, 18, 18, 17, 17, 17, 17, 17, 17, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
                16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 15, 15, 15, 15, 15, 15,
                15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
            ],
            75 => vec![
                0, 0, 0, 0, 0, 0, 0, 0, 0, 39, 40, 40, 40, 40, 40, 40, 40, 40, 40, 39, 39, 39, 38,
                38, 37, 37, 36, 36, 35, 34, 33, 33, 32, 31, 30, 30, 29, 28, 28, 27, 27, 26, 26, 25,
                25, 24, 24, 24, 23, 23, 23, 23, 22, 22, 22, 22, 22, 22, 21, 21, 21, 21, 21, 21, 21,
                21, 21, 21, 21, 21, 21, 21, 21, 21, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
                20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
            ],
            _ => panic!("Invalid p value"),
        },
    };

    let m = if fullmoves < v.len() as u32 {
        v[fullmoves as usize]
    } else {
        v[v.len() - 1]
    };

    m.max(2)
}
