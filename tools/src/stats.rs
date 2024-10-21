use crate::{method::Sample, plain_format::PlainReader};
use clap::Args;
use indicatif::{HumanBytes, ProgressBar, ProgressStyle};
use nn::feature_set::blocks::mobility;
use nn::feature_set::build::build_feature_set;
use shakmaty::{attacks, Bitboard, Color, File, Position, Rank, Role};
use std::hash::Hash;
use std::io::Write;
use std::{collections::HashMap, fs, io::BufWriter};

const FEATURE_SETS: [&str; 9] = ["all", "h", "v", "d1", "d2", "ph", "pv", "mb", "mc"];

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

    mobility: HashMap<(Role, usize), u64>,

    features_count: HashMap<String, u64>,
    features_adds_count: HashMap<String, u64>,
    features_rems_count: HashMap<String, u64>,
    features_updates_total: HashMap<String, u64>,
}

impl Stats {
    fn new() -> Self {
        Self {
            count: 0,
            files: HashMap::new(),
            ranks: HashMap::new(),
            mobility: HashMap::new(),
            features_count: HashMap::new(),
            features_adds_count: HashMap::new(),
            features_rems_count: HashMap::new(),
            features_updates_total: HashMap::new(),
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

        let occupied = board.occupied();
        let unoccupied = !occupied;

        let mut per_type = [0; 6];

        for (piece_square, piece) in board.clone().into_iter() {
            if piece.color == Color::White {
                per_type[piece.role as usize - 1] +=
                    mobility::mobility(board, piece_square).count();
            }
        }

        for (role, count) in per_type.iter().enumerate() {
            if role == Role::Pawn as usize - 1 && *count >= 15 {
                println!(
                    "Very high mobility {} FEN: {}",
                    *count,
                    shakmaty::fen::Fen(
                        sample
                            .position
                            .clone()
                            .into_setup(shakmaty::EnPassantMode::Always)
                    )
                );

                for (piece_square, piece) in board.clone().into_iter() {
                    if piece.color == Color::White && piece.role == Role::Pawn {
                        let attack = attacks::attacks(piece_square, piece, occupied);
                        let accesible = unoccupied.with(board.by_color(piece.color.other()));

                        println!("{:?}", (attack & accesible));
                    }
                }
            }
            *self.mobility.entry((Role::ALL[role], *count)).or_default() += 1;
        }

        for name in FEATURE_SETS {
            let mut features = vec![];
            let fs = build_feature_set(name);

            // count features
            fs.active_features(
                sample.position.board(),
                sample.position.turn(),
                Color::White,
                &mut features,
            );
            let mut counts = Vec::new();
            counts.resize(fs.num_features() as usize, 0);
            for f in features.clone() {
                counts[f as usize] += 1;
            }

            *self.features_count.entry(name.to_owned()).or_default() += features.len() as u64;

            for m in sample.position.legal_moves() {
                // count changed features
                let mut add_feats = vec![];
                let mut rem_feats = vec![];

                fs.changed_features(
                    sample.position.board(),
                    &m,
                    sample.position.turn(),
                    sample.position.turn(),
                    &mut add_feats,
                    &mut rem_feats,
                );

                let mut counts = counts.clone();
                let mut added_rows = vec![];
                let mut removed_rows = vec![];

                for &f in add_feats.iter() {
                    if counts[f as usize] == 0 {
                        added_rows.push(f);
                    }
                    counts[f as usize] += 1;
                }
                for &f in rem_feats.iter() {
                    counts[f as usize] -= 1;
                    if counts[f as usize] == 0 {
                        removed_rows.push(f);
                    }
                }

                while !added_rows.is_empty() {
                    if added_rows.last() == removed_rows.last() {
                        added_rows.pop();
                        removed_rows.pop();
                    } else if added_rows.first() == removed_rows.first() {
                        added_rows.swap_remove(0);
                        removed_rows.swap_remove(0);
                    } else {
                        break;
                    }
                }

                added_rows.sort();
                removed_rows.sort();

                // take advantage on the fact that features are sorted
                let mut i: i32 = added_rows.len() as i32 - 1;
                let mut j: i32 = removed_rows.len() as i32 - 1;

                while i >= 0 && j >= 0 {
                    if added_rows[i as usize] == removed_rows[j as usize] {
                        added_rows.swap_remove(i as usize);
                        removed_rows.swap_remove(j as usize);
                        i -= 1;
                        j -= 1;
                    } else if added_rows[i as usize] > removed_rows[j as usize] {
                        i -= 1;
                    } else {
                        j -= 1;
                    }
                }

                *self.features_adds_count.entry(name.to_owned()).or_default() +=
                    added_rows.len() as u64;
                *self.features_rems_count.entry(name.to_owned()).or_default() +=
                    removed_rows.len() as u64;

                *self
                    .features_updates_total
                    .entry(name.to_owned())
                    .or_default() += 1;
            }
        }
    }

    fn save(&self) -> std::io::Result<()> {
        let file = fs::File::create("stats.txt").expect("can't create stats file");
        let mut writer = BufWriter::new(&file);

        let mut files: Vec<_> = self.files.iter().collect();
        let mut ranks: Vec<_> = self.ranks.iter().collect();

        files.sort_by_key(|(_, count)| -(**count as i64));
        ranks.sort_by_key(|(_, count)| -(**count as i64));

        writeln!(writer, "Features:")?;
        for name in FEATURE_SETS {
            writeln!(
                writer,
                "{} {} {} {}",
                name,
                (*self.features_count.get(name).unwrap_or(&0) as f64) / self.count as f64,
                (*self.features_adds_count.get(name).unwrap_or(&0) as f64)
                    / (*self.features_updates_total.get(name).unwrap_or(&0) as f64),
                (*self.features_rems_count.get(name).unwrap_or(&0) as f64)
                    / (*self.features_updates_total.get(name).unwrap_or(&0) as f64),
            )?;
        }

        writeln!(writer, "Files:")?;
        for ((file, p), count) in &files {
            writeln!(writer, "{:?} {:?} {}", file, p, count)?;
        }

        writeln!(writer, "Ranks:")?;
        for ((rank, p), count) in &ranks {
            writeln!(writer, "{:?} {:?} {}", rank, p, count)?;
        }

        writeln!(writer, "Mobility:")?;
        let mut sorted = self
            .mobility
            .iter()
            .map(|((role, k), count)| (role, *k, *count))
            .collect::<Vec<_>>();
        sorted.sort();
        for (role, k, count) in sorted {
            writeln!(writer, "{:?} {} {}", role, k, count)?;
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

            if bar.position() % 100_000 == 0 {
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
