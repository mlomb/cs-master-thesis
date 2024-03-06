use crate::game_visitor::GameVisitor;
use crate::game_visitor::VisitorConfig;
use crate::method::eval::Eval;
use crate::method::eval::EvalArgs;
use crate::method::pqr::PQR;
use crate::method::stats_topk::StatsTopK;
use crate::method::WriteSample;
use clap::Args;
use clap::Subcommand;
use indicatif::{HumanCount, ProgressBar, ProgressStyle};
use pgn_reader::BufferedReader;
use std::error::Error;
use std::fs::File;
use std::io;
use std::io::BufWriter;
use zstd::Encoder;

#[derive(Args)]
pub struct BuildDatasetCommand {
    /// Path or URL of a .pgn or .pgn.zst file to read games
    #[arg(long, value_name = "input")]
    input: String,

    /// Output .csv (or csv.zst) file to write the samples
    #[arg(long, value_name = "output")]
    output: String,

    /// Whether to compress the output CSV with the ZSTD algorithm
    #[arg(long, default_value = "false")]
    compress: bool,

    /// Game visitor configuration
    #[clap(flatten)]
    visitor_config: VisitorConfig,

    /// Method to use
    #[command(subcommand)]
    subcommand: MethodSubcommand,
}

#[derive(Subcommand)]
pub enum MethodSubcommand {
    /// Generates a dataset with two columns: FEN string of a position, the observed move for the position.
    PQR,
    /// Generates a dataset with three columns: FEN string of a position, its score and the best move (both given by the engine)
    Eval(EvalArgs),
    /// Extracts statistics to build the TopK feature set
    StatsTopK,
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

    // compress output if asked
    let output_file = File::create(cmd.output.clone())?;
    let mut writer: Box<dyn io::Write> = if cmd.compress {
        // the encoder is buffered internally
        Box::new(Encoder::new(output_file, 3)?.auto_finish())
    } else {
        Box::new(BufWriter::new(output_file))
    };

    println!("Input: {}", cmd.input);
    println!("Output: {}", cmd.output);
    println!("Write compressed: {}", cmd.compress);

    let mut visitor = GameVisitor::new(cmd.visitor_config);
    let mut game_reader = BufferedReader::new(reader);
    let mut count = 0;

    let mut method: Box<dyn WriteSample> = match &cmd.subcommand {
        MethodSubcommand::PQR => Box::new(PQR {}),
        MethodSubcommand::Eval(args) => Box::new(Eval::new(args.clone())),
        MethodSubcommand::StatsTopK => Box::new(StatsTopK::new()),
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

            method.write_sample(&mut writer, &positions)?;

            count += 1;
            bar.set_message(format!("[Samples {}]", HumanCount(count).to_string()));
        }
    }
    bar.finish();

    println!("Done. Accepted positions: {}", count);

    Ok(())
}
