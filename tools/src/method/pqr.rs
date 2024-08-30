use super::{Sample, SampleEncoder};
use crate::pos_encoding::{encode_position, encoded_size};
use nn::feature_set::FeatureSet;
use rand::seq::SliceRandom;
use shakmaty::Position;
use std::io::Write;

pub struct PQREncoding;

impl SampleEncoder for PQREncoding {
    fn x_size(&self, feature_set: &Box<dyn FeatureSet>) -> usize {
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
        feature_set: &Box<dyn FeatureSet>,
    ) {
        let mut rng = rand::thread_rng();

        // P: initial
        let p_position = &sample.position;
        let moves = p_position.legal_moves();

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

        encode_position(&p_position, feature_set, write_x);
        encode_position(&q_position, feature_set, write_x);
        encode_position(&r_position, feature_set, write_x);
    }
}
