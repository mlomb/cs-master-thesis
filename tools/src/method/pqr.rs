use super::SampleEncoder;
use crate::encode::encoded_size;
use nn::feature_set::FeatureSet;
use std::io::Write;

pub struct PQR;

impl SampleEncoder for PQR {
    fn x_size(&self, feature_set: &Box<dyn FeatureSet>) -> usize {
        encoded_size(feature_set) * 3
    }

    fn y_size(&self) -> usize {
        0
    }

    fn write_sample(
        &self,
        sample: &super::Sample,
        write_x: &mut dyn Write,
        write_y: &mut dyn Write,
        feature_set: &Box<dyn FeatureSet>,
    ) {
    }

    /*
    fn read_sample(
        &self,
        read: &mut dyn BufRead,
        write_x: &mut dyn Write,
        _write_y: &mut dyn Write,
        feature_set: &Box<dyn FeatureSet>,
    ) {
        let mut rng = rand::thread_rng();
        let mut p_fen_bytes = Vec::with_capacity(128);
        let mut q_move_bytes = Vec::with_capacity(16);

        read.read_until(b',', &mut p_fen_bytes).unwrap();
        read.read_until(b'\n', &mut q_move_bytes).unwrap();

        // remove trailing comma and newline
        p_fen_bytes.pop();
        if q_move_bytes.last() == Some(&b'\n') {
            // it may not be present if EOF
            q_move_bytes.pop();
        }

        let p_fen = Fen::from_ascii(p_fen_bytes.as_slice()).unwrap();
        let p_position: Chess = p_fen.into_position(CastlingMode::Standard).unwrap();
        let moves = p_position.legal_moves();

        let q_move = UciMove::from_ascii(q_move_bytes.as_slice())
            .unwrap()
            .to_move(&p_position)
            .unwrap();
        let q_position = p_position.clone().play(&q_move).unwrap();

        // find a random r position
        let r_position = loop {
            let r_move = moves.choose(&mut rng).unwrap().clone();
            if r_move == q_move {
                // r != q
                continue;
            }

            break p_position.clone().play(&r_move).unwrap();
        };

        encode_position(feature_set, &p_position, write_x);
        encode_position(feature_set, &q_position, write_x);
        encode_position(feature_set, &r_position, write_x);
    }
    */
}
