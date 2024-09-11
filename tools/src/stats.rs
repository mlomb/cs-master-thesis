use crate::{method::Sample, plain_format::PlainReader};
use clap::Args;
use indicatif::{HumanBytes, ProgressBar, ProgressStyle};
use shakmaty::{Bitboard, Color, File, Position, Rank, Role};
use std::io::Write;
use std::{collections::HashMap, fs, io::BufWriter};

#[derive(Args)]
pub struct StatsCommand {
    /// Input to gather stats on
    #[arg(long, required = true)]
    input: String,
}

struct Stats {
    count: u64,

    files: HashMap<(File, ((Role, Color), (Role, Color))), u64>,
    ranks: HashMap<(Rank, ((Role, Color), (Role, Color))), u64>,
}

impl Stats {
    fn new() -> Self {
        Self {
            count: 0,
            files: HashMap::new(),
            ranks: HashMap::new(),
        }
    }

    fn add(&mut self, sample: Sample) {
        self.count += 1;

        let board = sample.position.board();

        for file in File::ALL.iter() {
            board
                .occupied()
                .intersect(Bitboard::from_file(file.clone()))
                .into_iter()
                .map_windows(|[l, r]| {
                    (
                        (
                            board.role_at(l.clone()).unwrap(),
                            board.color_at(l.clone()).unwrap(),
                        ),
                        (
                            board.role_at(r.clone()).unwrap(),
                            board.color_at(r.clone()).unwrap(),
                        ),
                    )
                })
                .for_each(|p| *self.files.entry((file.clone(), p)).or_default() += 1);
        }
        for rank in Rank::ALL.iter() {
            board
                .occupied()
                .intersect(Bitboard::from_rank(rank.clone()))
                .into_iter()
                .map_windows(|[l, r]| {
                    (
                        (
                            board.role_at(l.clone()).unwrap(),
                            board.color_at(l.clone()).unwrap(),
                        ),
                        (
                            board.role_at(r.clone()).unwrap(),
                            board.color_at(r.clone()).unwrap(),
                        ),
                    )
                })
                .for_each(|p| *self.ranks.entry((rank.clone(), p)).or_default() += 1);
        }
    }

    fn save(&self) -> std::io::Result<()> {
        let file = fs::File::create("stats.txt").expect("can't create stats file");
        let mut writer = BufWriter::new(&file);

        let mut files: Vec<_> = self.files.iter().collect();
        let mut ranks: Vec<_> = self.ranks.iter().collect();

        files.sort_by_key(|(_, count)| -(**count as i64));
        ranks.sort_by_key(|(_, count)| -(**count as i64));

        writeln!(writer, "Files:")?;
        for ((file, p), count) in &files {
            writeln!(writer, "{:?} {:?} {}", file, p, count)?;
        }

        writeln!(writer, "Ranks:")?;
        for ((rank, p), count) in &ranks {
            writeln!(writer, "{:?} {:?} {}", rank, p, count)?;
        }

        Ok(())
    }
}

pub fn stats(cmd: StatsCommand) {
    println!("Gathering stats of: {}", cmd.input);

    let mut reader = PlainReader::open(&cmd.input).expect("can't open input file");
    let mut stats = Stats::new();

    let bar = ProgressBar::new_spinner()
        .with_style(ProgressStyle::default_spinner()
        .template(
            "{spinner:.green} [Elapsed {elapsed_precise}] [Positions {human_pos} @ {per_sec}] {msg}",
        )
        .unwrap());

    while let Ok(Some(samples)) = reader.read_samples_line() {
        bar.inc(samples.len() as u64);

        for sample in samples {
            stats.add(sample);

            if bar.position() % 1_000_000 == 0 {
                stats.save().expect("can't save stats");

                bar.set_message(format!(
                    "[Read {}]",
                    HumanBytes(reader.bytes_read().unwrap()),
                ));
            }
        }
    }

    bar.finish();
}
