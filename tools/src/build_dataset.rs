use crate::game_visitor::GameVisitor;
use crate::game_visitor::VisitorConfig;
use crate::method::eval::Eval;
use crate::method::eval::EvalArgs;
use crate::method::pqr::PQR;
use crate::method::TrainingMethod;
use clap::Args;
use clap::Subcommand;
use indicatif::{HumanCount, ProgressBar, ProgressStyle};
use pgn_reader::BufferedReader;
use std::error::Error;
use std::fs::File;
use std::io;

#[derive(Args)]
pub struct BuildDatasetCommand {
    /// Path or URL of a .pgn or .pgn.zst file to read games
    #[arg(long, value_name = "input")]
    input: String,

    /// Output CSV file to write the samples
    #[arg(long, value_name = "output")]
    output: String,

    /// Game visitor configuration
    #[clap(flatten)]
    visitor_config: VisitorConfig,

    /// Training method to use
    #[command(subcommand)]
    subcommand: TrainMethodSubcommand,
}

#[derive(Subcommand)]
pub enum TrainMethodSubcommand {
    /// Generates a dataset with three columns: FEN strings of the positions P, Q and R.
    /// Given a transition P → Q in a game, R is selected from a legal move from P while R != Q.
    PQR,
    /// Generates a dataset with three columns: FEN string of a position, its score and the best move (both given by the engine)
    Eval(EvalArgs),
}

pub fn build_dataset(cmd: BuildDatasetCommand) -> Result<(), Box<dyn Error>> {
    // raw data stream (may be compressed)
    let raw_reader: Box<dyn io::Read> = if cmd.input.starts_with("http") {
        Box::new(reqwest::blocking::get(cmd.input.clone())?)
    } else {
        Box::new(File::open(&cmd.input)?)
    };

    // decompress if necessary
    let reader: Box<dyn io::Read> = if cmd.input.ends_with(".zst") {
        Box::new(zstd::Decoder::new(raw_reader)?)
    } else {
        raw_reader
    };

    println!("Input: {}", cmd.input);
    println!("Output: {}", cmd.output);

    //let engine = UciEngine::new(cmd.);
    let mut visitor = GameVisitor::new(cmd.visitor_config);
    let mut game_reader = BufferedReader::new(reader);
    let mut out_file = File::create(cmd.output)?;
    let mut count = 0;

    let mut method: Box<dyn TrainingMethod> = match &cmd.subcommand {
        TrainMethodSubcommand::PQR => Box::new(PQR::new()),
        TrainMethodSubcommand::Eval(args) => Box::new(Eval::new(args.clone())),
    };

    let bar = ProgressBar::new_spinner()
        .with_style(ProgressStyle::default_spinner()
        .template(
            "{spinner:.green} [Elapsed {elapsed_precise}] [Games {human_pos} @ {per_sec}] {msg}",
        )
        .unwrap());

    while let Ok(Some(sample)) = game_reader.read_game(&mut visitor) {
        bar.inc(1);

        if let Some(positions) = sample {
            assert!(positions.len() > 0);

            method.write_sample(&mut out_file, &positions)?;

            count += 1;
            bar.set_message(format!("[Samples {}]", HumanCount(count).to_string()));
        }
    }
    bar.finish();

    println!("Done. Accepted positions: {}", count);

    Ok(())
}
