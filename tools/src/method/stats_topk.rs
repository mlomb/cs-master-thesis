use super::Method;
use rand::{seq::SliceRandom, Rng};
use shakmaty::{Chess, Color, Position};
use std::{
    fs::File,
    io::{self, Write},
};

const MAX_TUPLES: usize = 64 * 6 * 64 * 6 * 2;

pub struct StatsTopK {
    total: u32,
    counts: [u32; MAX_TUPLES],
}

impl StatsTopK {
    pub fn new() -> Self {
        StatsTopK {
            total: 0,
            counts: [0; MAX_TUPLES],
        }
    }
}

impl Method for StatsTopK {
    fn write_sample(&mut self, file: &mut File, positions: &Vec<Chess>) -> io::Result<()> {
        let mut rng = rand::thread_rng();
        let mut tries = 0;

        while tries < 10 {
            tries += 1;

            // choose random position
            let position = positions.choose(&mut rng).unwrap().clone();
            if position.turn() != Color::White {
                // only consider positions where it's white's turn
                continue;
            }

            let board = position.board();

            for (square1, piece1) in board.clone().into_iter() {
                if piece1.color == Color::Black {
                    continue;
                }

                for (square2, piece2) in board.clone().into_iter() {
                    if square1 == square2 {
                        continue;
                    }

                    let index = (square1 as usize * 6 * 64 * 6 * 2)
                        + (piece1.role as usize * 64 * 6 * 2)
                        + (square2 as usize * 6 * 2)
                        + (piece2.role as usize * 2)
                        + (piece2.color as usize);

                    self.counts[index] += 1;
                }
            }
            self.total += 1;

            // only consider one position per game
            break;
        }

        // write file
        if self.total % 100 == 0 {
            file.set_len(0)?; // reset file

            for square1 in 0..64 {
                for piece1 in 0..6 {
                    for square2 in 0..64 {
                        for piece2 in 0..6 {
                            for color in 0..2 {
                                let index = (square1 * 6 * 64 * 6 * 2)
                                    + (piece1 * 64 * 6 * 2)
                                    + (square2 * 6 * 2)
                                    + (piece2 * 2)
                                    + color;

                                writeln!(
                                    file,
                                    "{},{},{},{},{},{}",
                                    square1, piece1, square2, piece2, color, self.counts[index]
                                )?;
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }
}
