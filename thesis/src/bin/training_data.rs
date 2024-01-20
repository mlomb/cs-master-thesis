use indicatif::{ProgressBar, ProgressStyle};
use pgn_reader::{BufferedReader, RawHeader, SanPlus, Skip, Visitor};
use rand::prelude::SliceRandom;
use rand::Rng;
use shakmaty::{Board, Chess, Color, Outcome, Position};
use std::{
    cell::RefCell,
    fs::File,
    io::{self},
    rc::Rc,
};

#[derive(Debug)]
struct Sample {
    parent: Board,
    observed: Board,
    random: Board,
}

#[derive(Debug)]
struct GameVisitor {
    event: String,
    termination: String,
    white_elo: i32,
    black_elo: i32,

    positions: Vec<Chess>,
}

impl GameVisitor {
    fn new() -> Self {
        GameVisitor {
            event: "".to_string(),
            termination: "".to_string(),
            white_elo: 0,
            black_elo: 0,
            positions: vec![Chess::default()],
        }
    }
}

impl Visitor for GameVisitor {
    type Result = Option<Sample>;

    fn begin_game(&mut self) {
        self.event = "".to_string();
        self.termination = "".to_string();
        self.white_elo = 0;
        self.black_elo = 0;
        self.positions.truncate(1); // only keep starting board
    }

    fn header(&mut self, _key: &[u8], _value: RawHeader<'_>) {
        let key = String::from_utf8_lossy(_key);
        let value = String::from_utf8_lossy(_value.as_bytes());

        if key == "Event" {
            self.event = value.to_string();
        } else if key == "Termination" {
            self.termination = value.to_string();
        } else if key == "BlackElo" {
            self.black_elo = value.parse().unwrap_or(0);
        } else if key == "WhiteElo" {
            self.white_elo = value.parse().unwrap_or(0);
        }
    }

    fn end_headers(&mut self) -> Skip {
        const ELO_THRESHOLD: i32 = 2000;

        let keep =
            // keep rated classical games
            self.event.starts_with("Rated Classical") &&
            // keep normal terminations (excl. Abandoned, Time forfeit, Rules infraction)
            self.termination == "Normal" &&
            // keep games with good elo
            self.white_elo >= ELO_THRESHOLD &&
            self.black_elo >= ELO_THRESHOLD;

        Skip(!keep)
    }

    fn begin_variation(&mut self) -> Skip {
        Skip(true)
    }

    fn san(&mut self, _san_plus: SanPlus) {
        let pos = self.positions.last().unwrap().clone();
        let mov = _san_plus.san.to_move(&pos).unwrap();
        self.positions.push(pos.play(&mov).unwrap());
    }

    fn end_game(&mut self) -> Self::Result {
        if self.positions.len() <= 5 {
            // note: skipped games go through here too
            return None;
        }

        let mut rng = rand::thread_rng();

        loop {
            let index: usize = rng.gen_range(0..self.positions.len() - 1); // dont pick last
            let parent = &self.positions[index];
            let observed = &self.positions[index + 1];
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

            let sample: Sample = Sample {
                parent: parent.board().clone(),
                observed: observed.board().clone(),
                random: random.board().clone(),
            };

            return Some(sample);
        }
    }
}

fn main() -> io::Result<()> {
    let files = vec![
        "../data/source/lichess_db_standard_rated_2023-09.pgn.zst",
        "../data/source/lichess_db_standard_rated_2023-10.pgn.zst",
        "../data/source/lichess_db_standard_rated_2023-11.pgn.zst",
    ];

    let bar_style = ProgressStyle::default_spinner()
        .template(
            "{spinner:.green} [Elapsed {elapsed_precise}] [Parsed {human_pos} @ {per_sec}] {msg} {prefix}",
        )
        .unwrap();

    for path in files {
        let file = File::open(&path)?;
        let uncompressed: Box<dyn io::Read> = if path.ends_with(".zst") {
            Box::new(zstd::Decoder::new(file)?)
        } else {
            Box::new(file)
        };

        let bar = ProgressBar::new_spinner()
            .with_style(bar_style.clone())
            .with_prefix(path);

        let mut reader = BufferedReader::new(uncompressed);
        let mut visitor = GameVisitor::new();
        let mut count = 0;

        while let Ok(Some(sample)) = reader.read_game(&mut visitor) {
            bar.inc(1);

            if let Some(sample) = sample {
                println!("{:?}", sample);
                count += 1;
                bar.set_message(format!("[Accepted {}]", count));
            }
        }
        bar.finish();
    }

    Ok(())
}
