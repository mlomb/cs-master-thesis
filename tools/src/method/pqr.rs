use super::Method;
use rand::{seq::SliceRandom, Rng};
use shakmaty::{fen::Fen, Chess, EnPassantMode, Position};
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

impl Method for PQR {
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
