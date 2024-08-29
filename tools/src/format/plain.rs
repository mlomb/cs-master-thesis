use crate::method::Sample;
use memmap2::{Mmap, MmapOptions};
use shakmaty::{fen::Fen, uci::UciMove, CastlingMode, Chess, EnPassantMode, Move, Position};
use std::io::{self, Seek, Write};
use std::{
    fs::File,
    io::{BufRead, BufWriter, Cursor},
};

/// Read files in .plain and .plainpack formats
pub struct PlainReader {
    reader: Cursor<&'static [u8]>,

    // Must be kept alive for the lifetime of the reader
    #[allow(dead_code)]
    mmap: Mmap,
}

impl PlainReader {
    pub fn new(file: File) -> PlainReader {
        let mmap = unsafe { MmapOptions::new().map(&file).expect("can't mmap file") };
        let slice: &'static [u8] = unsafe { std::mem::transmute(mmap.as_ref()) };
        let reader = Cursor::new(slice);

        PlainReader { mmap, reader }
    }

    /// Read a line of samples from the file
    /// The format is: `fen,score,bestmove(,actualmove,score,bestmove)*`
    pub fn read_samples_line(&mut self) -> io::Result<Option<Vec<Sample>>> {
        let mut samples = Vec::new();
        let mut line = String::new();
        let mut fen_bytes = Vec::with_capacity(128);
        let mut score_bytes = Vec::with_capacity(16);
        let mut move_bytes = Vec::with_capacity(16);

        // read whole line
        self.reader.read_line(&mut line)?;
        if line.ends_with('\n') {
            line.pop(); // remove trailing newline
        }
        if line.is_empty() {
            return Ok(None);
        }

        let mut cursor = Cursor::new(line.as_bytes());

        // parse the FEN into a position
        let mut position: Chess = {
            cursor.read_until(b',', &mut fen_bytes)?;
            fen_bytes.pop(); // remove trailing comma

            Fen::from_ascii(fen_bytes.as_slice())
                .unwrap()
                .into_position(CastlingMode::Standard)
                .unwrap()
        };

        loop {
            score_bytes.clear();
            move_bytes.clear();

            cursor.read_until(b',', &mut score_bytes)?;
            cursor.read_until(b',', &mut move_bytes)?;

            if score_bytes.is_empty() || move_bytes.is_empty() {
                break;
            }

            // remove trailing comma
            score_bytes.pop();
            if move_bytes.last() == Some(&b',') {
                move_bytes.pop();
            }

            let score_str = String::from_utf8_lossy(&score_bytes);
            let best_move_uci: UciMove = UciMove::from_ascii(move_bytes.as_slice()).unwrap();

            let score = score_str.parse::<i32>().unwrap();
            let bestmove = best_move_uci.to_move(&position).unwrap();

            samples.push(Sample {
                position: position.clone(),
                bestmove: bestmove.clone(),
                score,
            });

            if cursor.position() < line.len() as u64 {
                // read actual move to mutate position
                move_bytes.clear();
                cursor.read_until(b',', &mut move_bytes)?;
                move_bytes.pop(); // remove trailing comma

                let actualmove = if move_bytes.len() > 0 {
                    UciMove::from_ascii(move_bytes.as_slice())
                        .unwrap()
                        .to_move(&position)
                        .unwrap()
                } else {
                    bestmove
                };

                // play the move to get the next position
                position = position.play(&actualmove).unwrap();
            }
        }

        Ok(Some(samples))
    }

    pub fn bytes_read(&self) -> u64 {
        self.reader.position()
    }
}

pub struct PlainWriter {
    is_first: bool,
    writer: BufWriter<File>,

    last_pos: Option<Chess>,
    last_bestmove: Option<Move>,
}

impl PlainWriter {
    pub fn new(file: File) -> PlainWriter {
        PlainWriter {
            is_first: true,
            writer: BufWriter::new(file),

            last_pos: None,
            last_bestmove: None,
        }
    }

    pub fn write_sample(&mut self, sample: Sample) -> io::Result<()> {
        // check if a move from the last position matches the sample to write
        let mut chain_with = None;

        if let Some(ref last_pos) = self.last_pos {
            for mov in last_pos.legal_moves() {
                let mut moved_pos = last_pos.clone();
                moved_pos.play_unchecked(&mov);

                if moved_pos == sample.position {
                    // match, we can skip the fen!
                    chain_with = Some(mov);
                }
            }
        }

        if let Some(actualmove) = chain_with {
            // write `,actualmove,`
            if self.last_bestmove.as_ref() == Some(&actualmove) {
                // if it matches, skip it
                write!(self.writer, ",,")?;
            } else {
                write!(
                    self.writer,
                    ",{},",
                    UciMove::from_move(&actualmove, CastlingMode::Standard)
                )?
            }
        } else {
            // write `\nfen,`
            let fen = Fen(sample.position.clone().into_setup(EnPassantMode::Always)).to_string();

            if !self.is_first {
                // only write newline if it is not the first sample in the file
                write!(self.writer, "\n")?;
            }
            self.is_first = false;

            write!(self.writer, "{},", fen)?;
        }

        // write `score,bestmove`
        write!(
            self.writer,
            "{},{}",
            sample.score,
            UciMove::from_move(&sample.bestmove, CastlingMode::Standard)
        )?;

        // store last
        self.last_pos = Some(sample.position);
        self.last_bestmove = Some(sample.bestmove);

        Ok(())
    }

    pub fn bytes_written(&mut self) -> io::Result<u64> {
        self.writer.stream_position()
    }
}
