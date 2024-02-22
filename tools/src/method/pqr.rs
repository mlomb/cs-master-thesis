use super::{ReadSample, WriteSample};
use nn::feature_set::FeatureSet;
use rand::{seq::SliceRandom, Rng};
use shakmaty::CastlingMode;
use shakmaty::{fen::Fen, Chess, EnPassantMode, Position};
use std::io::BufRead;
use std::io::BufReader;
use std::{
    fs::File,
    io::{self, Write},
};
pub struct PQR;

impl PQR {
    pub fn new() -> Self {
        PQR
    }
}

impl WriteSample for PQR {
    fn write_sample(&mut self, file: &mut File, positions: &Vec<Chess>) -> io::Result<()> {
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
                file,
                "{},{},{}",
                Fen(parent.clone().into_setup(EnPassantMode::Legal)),
                Fen(observed.clone().into_setup(EnPassantMode::Legal)),
                Fen(random.into_setup(EnPassantMode::Legal))
            );
        }
    }

    /*
           //  flip boards to be from white's POV
           if (index % 2) == 0 {
               // W B B
               observed.flip_vertical();
               observed.swap_colors();
               random.flip_vertical();
               random.swap_colors();
           } else {
               // B W W
               parent.flip_vertical();
               parent.swap_colors();
           }
    */
}

impl ReadSample for PQR {
    fn read_sample(
        &mut self,
        read: &mut BufReader<File>,
        buffer: &mut [u64],
        feature_set: &Box<dyn FeatureSet>,
    ) -> usize {
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

        // buffer offset etc
        feature_set.write_inputs(&p_position, buffer);
        feature_set.write_inputs(&q_position, buffer);
        feature_set.write_inputs(&r_position, buffer);

        todo!()
    }
}
