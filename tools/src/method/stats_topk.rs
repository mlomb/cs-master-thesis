use super::WriteSample;
use rand::seq::SliceRandom;
use shakmaty::{Chess, Color, Position, Role, Square};
use std::{
    fs::File,
    io::{self, BufWriter, Seek, Write},
};

struct Tuple(Square, Role, Square, Role, Color);

impl Tuple {
    fn index(&self) -> usize {
        // reference piece
        let square1 = self.0;
        let role1 = self.1;

        // board piece
        let square2 = self.2;
        let role2 = self.3;
        let color2 = self.4;

        (square1 as usize * 6 * 64 * 6 * 2)
            + ((role1 as usize - 1) * 64 * 6 * 2)
            + (square2 as usize * 6 * 2)
            + ((role2 as usize - 1) * 2)
            + (color2 as usize)
    }
}

pub struct StatsTopK {
    total: u32,
    counts: Vec<u32>,
    tuples: Vec<Tuple>,
}

impl StatsTopK {
    pub fn new() -> Self {
        let mut tuples = Vec::new();

        for square1 in Square::ALL {
            for role1 in Role::ALL {
                for square2 in Square::ALL {
                    for role2 in Role::ALL {
                        for color2 in Color::ALL {
                            tuples.push(Tuple(square1, role1, square2, role2, color2));
                        }
                    }
                }
            }
        }

        StatsTopK {
            total: 0,
            counts: vec![0; tuples.len()],
            tuples,
        }
    }
}

impl WriteSample for StatsTopK {
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

                    let index =
                        Tuple(square1, piece1.role, square2, piece2.role, piece2.color).index();

                    self.counts[index] += 1;
                }
            }
            self.total += 1;

            // only consider one position per game
            break;
        }

        // write file
        if self.total % 100_000 == 0 {
            // reset file
            file.seek(io::SeekFrom::Start(0))?;
            file.set_len(0)?;

            let mut f = BufWriter::new(file);

            self.tuples
                .sort_by(|a, b| self.counts[b.index()].cmp(&self.counts[a.index()]));

            for t in &self.tuples {
                if self.counts[t.index()] == 0 {
                    // skip zero counts
                    continue;
                }

                writeln!(
                    f,
                    "{:?},{:?},{:?},{:?},{:?},{}",
                    t.0,
                    t.1,
                    t.2,
                    t.3,
                    t.4,
                    self.counts[t.index()]
                )?;
            }

            f.flush()?;
        }

        Ok(())
    }
}
