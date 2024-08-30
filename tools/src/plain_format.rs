use crate::method::Sample;
use memmap2::{Mmap, MmapOptions};
use shakmaty::{fen::Fen, uci::UciMove, CastlingMode, EnPassantMode, Position};
use shakmaty::{Chess, Move};
use std::fs::File;
use std::io::{self, BufWriter, Read, Seek, Write};
use std::io::{BufRead, Cursor};
use std::path::Path;

/// Read files in .plain format
pub struct PlainReader<R: Read> {
    reader: R,

    /// Must be kept alive for the lifetime of the reader
    #[allow(dead_code)]
    mmap: Option<Mmap>,
}

impl<'a> PlainReader<Cursor<&'a [u8]>> {
    pub fn open_with_limits<P>(
        path: P,
        offset: u64,
        length: u64,
    ) -> io::Result<PlainReader<Cursor<&'a [u8]>>>
    where
        P: AsRef<Path>,
    {
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file).expect("can't mmap file") };
        let slice: &'a [u8] = unsafe { std::mem::transmute(mmap.as_ref()) };
        let subslice = if length > 0 {
            &slice[offset as usize..(offset + length) as usize]
        } else {
            &slice[offset as usize..]
        };

        let mut reader = Cursor::new(subslice);

        if offset > 0 {
            // skip until the next valid line
            reader.skip_until(b'\n')?;
        }

        Ok(PlainReader {
            reader,
            mmap: Some(mmap),
        })
        // TODO: are we leaking the file?
    }

    pub fn open<P>(path: P) -> io::Result<PlainReader<Cursor<&'a [u8]>>>
    where
        P: AsRef<Path>,
    {
        Self::open_with_limits(path, 0, 0)
    }
}

impl<R: BufRead + Seek> PlainReader<R> {
    #[allow(dead_code)]
    fn new(reader: R) -> PlainReader<R> {
        PlainReader { reader, mmap: None }
    }

    pub fn read_samples_line(&mut self) -> io::Result<Option<Vec<Sample>>> {
        let mut samples = Vec::new();
        let mut line = String::new();
        let mut fen_bytes = Vec::with_capacity(128);
        let mut score_bytes = Vec::with_capacity(16);
        let mut move_bytes = Vec::with_capacity(16);

        // read whole line
        self.reader.read_line(&mut line).unwrap();
        if !line.ends_with('\n') || line.len() == 0 {
            // exit if there is no more to read
            // the last line may not have a newline
            // however we must account for the case where the sample is cut off
            // so we just skip it
            return Ok(None);
        }
        // remove trailing newline
        line.pop();

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

        // read list of move and scores
        loop {
            score_bytes.clear();
            move_bytes.clear();

            cursor.read_until(b',', &mut score_bytes)?;
            cursor.read_until(b',', &mut move_bytes)?;

            if score_bytes.is_empty() || move_bytes.is_empty() {
                break;
            }

            // remove trailing commas
            score_bytes.pop();
            if move_bytes.last() == Some(&b',') {
                move_bytes.pop();
            }

            let bestmove = UciMove::from_ascii(move_bytes.as_slice())
                .unwrap()
                .to_move(&position)
                .unwrap();

            let score = String::from_utf8_lossy(&score_bytes)
                .parse::<i32>()
                .unwrap();

            samples.push(Sample {
                position: position.clone(),
                bestmove: bestmove.clone(),
                score,
            });

            // if there is more, play the played move to get the next position
            if cursor.position() < line.len() as u64 {
                // read played move to mutate position
                move_bytes.clear();
                cursor.read_until(b',', &mut move_bytes)?;
                move_bytes.pop(); // remove trailing comma

                let playedmove = if move_bytes.len() > 0 {
                    UciMove::from_ascii(move_bytes.as_slice())
                        .unwrap()
                        .to_move(&position)
                        .unwrap()
                } else {
                    bestmove
                };

                // play the move to get the next position
                position = position.play(&playedmove).unwrap();
            }
        }

        Ok(Some(samples))
    }

    pub fn bytes_read(&mut self) -> io::Result<u64> {
        self.reader.stream_position()
    }
}

pub struct PlainWriter<W: Write + Seek> {
    writer: W,

    is_first: bool,
    last_pos: Option<Chess>,
    last_bestmove: Option<Move>,
}

impl<'a> PlainWriter<BufWriter<File>> {
    pub fn open<P>(path: P) -> io::Result<PlainWriter<BufWriter<File>>>
    where
        P: AsRef<Path>,
    {
        Ok(PlainWriter {
            writer: BufWriter::new(File::create(path)?),

            is_first: true,
            last_pos: None,
            last_bestmove: None,
        })
        // TODO: are we leaking the file?
    }
}

impl<W: Write + Seek> PlainWriter<W> {
    #[allow(dead_code)]
    pub fn new(writer: W) -> PlainWriter<W> {
        PlainWriter {
            writer,

            is_first: true,
            last_pos: None,
            last_bestmove: None,
        }
    }

    pub fn write_sample(&mut self, sample: &Sample) -> io::Result<()> {
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

        if let Some(playedmove) = chain_with {
            // write `,playedmove,`
            if self.last_bestmove.as_ref() == Some(&playedmove) {
                // if it matches, skip it
                write!(self.writer, ",,")?;
            } else {
                write!(
                    self.writer,
                    ",{},",
                    UciMove::from_move(&playedmove, CastlingMode::Standard)
                )?
            }
        } else {
            // write `\nfen,`

            if !self.is_first {
                // only write newline if it is not the first sample in the file
                write!(self.writer, "\n")?;
            }
            self.is_first = false;

            write!(
                self.writer,
                "{},",
                Fen(sample.position.clone().into_setup(EnPassantMode::Always))
            )?;
        }

        // write `score,bestmove`
        write!(
            self.writer,
            "{},{}",
            sample.score,
            UciMove::from_move(&sample.bestmove, CastlingMode::Standard)
        )?;

        // store last
        self.last_pos = Some(sample.position.clone());
        self.last_bestmove = Some(sample.bestmove.clone());

        Ok(())
    }

    pub fn finish(&mut self) -> io::Result<()> {
        writeln!(self.writer)?;
        self.writer.flush()
    }

    pub fn bytes_written(&mut self) -> io::Result<u64> {
        self.writer.stream_position()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use io::{BufReader, BufWriter};

    #[test]
    fn test_ignore_last() {
        let plain =
            "r7/6p1/2q1bpkp/1p1Np3/1P2P3/7P/2P1QPP1/3R2K1 b - - 6 24,-398,g6h7\nr7/6pk/2q1bp1p/1p";
        let mut reader = PlainReader::new(BufReader::new(Cursor::new(plain)));

        let res1 = reader.read_samples_line();
        let res2 = reader.read_samples_line();

        assert_eq!(res1.unwrap().unwrap().len(), 1);
        assert!(res2.unwrap().is_none());
    }

    #[test]
    fn test_individual() {
        for true_sample in test_samples() {
            let mut buffer = Vec::new();

            // write
            {
                let mut writer = PlainWriter::new(BufWriter::new(Cursor::new(&mut buffer)));
                writer.write_sample(&true_sample).unwrap();
                writer.finish().unwrap();
            }

            println!("{:?}", String::from_utf8_lossy(&buffer));

            // read
            let mut reader = PlainReader::new(BufReader::new(Cursor::new(buffer)));
            let actual_sample = reader.read_samples_line().unwrap().unwrap();

            assert_sample_eq(&true_sample, &actual_sample[0]);
        }
    }

    #[test]
    fn test_sequence() {
        let mut buffer = Vec::new();

        // write
        {
            let mut writer = PlainWriter::new(BufWriter::new(Cursor::new(&mut buffer)));
            for true_sample in test_samples() {
                writer.write_sample(&true_sample).unwrap();
            }
            writer.finish().unwrap();
        }

        println!("{:?}", String::from_utf8_lossy(&buffer));

        // read
        let mut reader = PlainReader::new(BufReader::new(Cursor::new(buffer)));
        let mut read_samples = Vec::new();

        while let Ok(Some(actual_sample)) = reader.read_samples_line() {
            read_samples.extend(actual_sample);
        }

        for (true_sample, actual_sample) in test_samples().iter().zip(read_samples.iter()) {
            assert_sample_eq(true_sample, actual_sample);
        }
    }

    fn assert_sample_eq(true_sample: &Sample, actual_sample: &Sample) {
        assert_eq!(
            // piece placement, turn, castling rights, en passant
            true_sample.position,
            actual_sample.position
        );
        assert_eq!(
            true_sample.position.halfmoves(),
            actual_sample.position.halfmoves()
        );
        assert_eq!(
            true_sample.position.fullmoves(),
            actual_sample.position.fullmoves()
        );
        assert_eq!(true_sample.score, actual_sample.score);
        assert_eq!(true_sample.bestmove, actual_sample.bestmove);
    }

    fn test_samples() -> Vec<Sample> {
        fn make_sample(fen: &str, bestmove: &str, score: i32) -> Sample {
            let position = Fen::from_ascii(fen.as_bytes())
                .unwrap()
                .into_position(CastlingMode::Standard)
                .unwrap();

            let bestmove = UciMove::from_ascii(bestmove.as_bytes())
                .unwrap()
                .to_move(&position)
                .unwrap();

            Sample {
                position,
                bestmove,
                score,
            }
        }

        vec![
            make_sample("2K5/p2P4/6k1/6n1/1P2P3/1P1N4/8/8 b - - 0 56", "g5e4", -1728),
            make_sample("8/3k4/1B1P2b1/5n2/8/P1K5/8/8 w - - 0 49", "b6c7", -7),
            make_sample("8/2Bk4/3P2b1/5n2/8/P1K5/8/8 b - - 1 49", "f5e3", 18),
            make_sample("8/2Bk4/3P2b1/8/8/P1K1n3/8/8 w - - 2 50", "c3d4", -17),
            make_sample("8/2Bk4/3P2b1/8/3K4/P3n3/8/8 b - - 3 50", "e3c2", 21),
            make_sample("8/2k5/p1P5/5r2/2K5/8/P7/7R b - - 4 39", "f5f7", -1),
            make_sample("8/3kn3/p7/3P3p/3K2p1/P5B1/6PP/8 b - - 2 31", "h5h4", -730),
            make_sample("8/5k2/1p5r/p1pPR3/P1P3K1/8/8/8 w - - 2 57", "g4g5", 212),
        ]
    }
}
