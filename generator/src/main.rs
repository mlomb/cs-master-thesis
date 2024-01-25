pub mod sample;
pub mod visitor;

use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use pgn_reader::BufferedReader;
use std::{
    fs::File,
    io::{self},
};
use visitor::GameVisitor;

#[derive(Parser)]
struct Cli {
    /// List of .pgn or .pgn.zst to read games
    inputs: Vec<String>,

    /// Output binary file to write the samples
    #[arg(short, long, value_name = "output")]
    output: String,

    /// Only accept games where both player have at least this elo
    #[arg(short, long, value_name = "min-elo")]
    min_elo: i32,
}

fn main() -> io::Result<()> {
    let args = Cli::parse_from(wild::args());

    println!(
        "Parsing {} files and writing to {}",
        args.inputs.len(),
        args.output
    );

    let bar_style = ProgressStyle::default_spinner()
        .template(
            "{spinner:.green} [Elapsed {elapsed_precise}] [Parsed {human_pos} @ {per_sec}] {msg} {prefix}",
        )
        .unwrap();

    let mut count = 0;
    let mut out_file = File::create(args.output)?;
    let mut visitor = GameVisitor::new(args.min_elo);

    for path in args.inputs {
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

        while let Ok(Some(sample)) = reader.read_game(&mut visitor) {
            bar.inc(1);

            if let Some(sample) = sample {
                sample.write_to(&mut out_file)?;

                count += 1;
                bar.set_message(format!("[Accepted {}]", count));
            }
        }
        bar.finish();
    }

    Ok(())
}
