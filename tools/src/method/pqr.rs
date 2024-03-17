use super::{encoded_size, ReadSample, WriteSample};
use nn::feature_set::FeatureSet;
use rand::{seq::SliceRandom, Rng};
use shakmaty::uci::Uci;
use shakmaty::CastlingMode;
use shakmaty::{fen::Fen, Chess, EnPassantMode, Position};
use std::io::BufRead;
use std::io::{self, Write};

pub struct PQR;

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

            for _move in parent.legal_moves() {
                if parent.clone().play(&_move).unwrap() == *observed {
                    return writeln!(
                        write,
                        "{},{}",
                        Fen(parent.clone().into_setup(EnPassantMode::Legal)),
                        _move.to_uci(CastlingMode::Standard)
                    );
                }
            }

            unreachable!("No legal move found");
        }
    }
}

impl ReadSample for PQR {
    fn x_size(&self, feature_set: &Box<dyn FeatureSet>) -> usize {
        encoded_size(feature_set) * 3
    }

    fn y_size(&self) -> usize {
        0
    }

    fn read_sample(
        &mut self,
        read: &mut dyn BufRead,
        write_x: &mut dyn Write,
        write_y: &mut dyn Write,
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

        let q_move = Uci::from_ascii(q_move_bytes.as_slice())
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

        //feature_set.encode(p_position.board(), p_position.turn(), write_x);
        //feature_set.encode(q_position.board(), q_position.turn(), write_x);
        //feature_set.encode(r_position.board(), r_position.turn(), write_x);
    }
}
