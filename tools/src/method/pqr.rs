use super::{ReadSample, WriteSample};
use nn::feature_set::FeatureSet;
use rand::{seq::SliceRandom, Rng};
use shakmaty::CastlingMode;
use shakmaty::{fen::Fen, Chess, EnPassantMode, Position};
use std::io::{self, Write};
use std::io::{BufRead, Cursor};
use std::io::{BufReader, Read};

pub struct PQR;

impl PQR {
    pub fn new() -> Self {
        PQR
    }
}

impl WriteSample for PQR {
    fn write_sample(&mut self, write: &mut dyn Write, positions: &Vec<Chess>) -> io::Result<()> {
        let mut rng = rand::thread_rng();

        loop {
            let index: usize = rng.gen_range(0..positions.len() - 1); // dont pick last
            let parent = &positions[index];
            let observed = &positions[index + 1];
            let moves = parent.legal_moves();

            if moves.len() <= 1 {
                // not enough moves to choose from
                // e.g check
                continue;
            }

            let random = loop {
                let mov = moves.choose(&mut rng).unwrap().clone();
                let pos = parent.clone().play(&mov).unwrap();
                if pos != *observed {
                    break pos;
                }
            };

            return writeln!(
                write,
                "{},{},{}",
                Fen(parent.clone().into_setup(EnPassantMode::Legal)),
                Fen(observed.clone().into_setup(EnPassantMode::Legal)),
                Fen(random.into_setup(EnPassantMode::Legal))
            );
        }
    }
}

impl ReadSample for PQR {
    fn sample_size(&self, feature_set: &Box<dyn FeatureSet>) -> usize {
        feature_set.encoded_size() * 3
    }

    fn read_sample(
        &mut self,
        read: &mut BufReader<Box<dyn Read>>,
        write: &mut Cursor<&mut [u8]>,
        feature_set: &Box<dyn FeatureSet>,
    ) {
        let mut p_bytes = Vec::with_capacity(128);
        let mut q_bytes = Vec::with_capacity(128);
        let mut r_bytes = Vec::with_capacity(128);

        read.read_until(b',', &mut p_bytes).unwrap();
        read.read_until(b',', &mut q_bytes).unwrap();
        read.read_until(b'\n', &mut r_bytes).unwrap();

        // remove trailing comma and newline
        p_bytes.pop();
        q_bytes.pop();
        r_bytes.pop();

        let p_fen = Fen::from_ascii(p_bytes.as_slice()).unwrap();
        let q_fen = Fen::from_ascii(q_bytes.as_slice()).unwrap();
        let r_fen = Fen::from_ascii(r_bytes.as_slice()).unwrap();

        let p_position: Chess = p_fen.into_position(CastlingMode::Standard).unwrap();
        let q_position: Chess = q_fen.into_position(CastlingMode::Standard).unwrap();
        let r_position: Chess = r_fen.into_position(CastlingMode::Standard).unwrap();

        feature_set.encode(p_position.board(), p_position.turn(), write);
        feature_set.encode(q_position.board(), q_position.turn(), write);
        feature_set.encode(r_position.board(), r_position.turn(), write);
    }
}
