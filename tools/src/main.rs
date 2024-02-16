mod build_dataset;
mod game_visitor;
mod uci_engine;

use build_dataset::TrainMethodCommand;
use clap::{arg, Parser, Subcommand, ValueEnum};
use game_visitor::VisitorConfig;
use nn::feature_set::FeatureSet;
use shakmaty::{fen::Fen, uci::Uci, CastlingMode, Chess, Position};
use std::{
    error::Error,
    fs::File,
    io::{BufRead, BufReader},
};

#[derive(ValueEnum, Clone)]
enum FeatureSetChoice {
    Basic,
    HalfKP,
}

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Builds a dataset for a speific training method using PGN files as input. Only one sample per game is extracted.
    BuildDataset {
        /// Path or URL of a .pgn or .pgn.zst file to read games
        #[arg(long, value_name = "input")]
        input: String,

        /// Output CSV file to write the samples
        #[arg(long, value_name = "output")]
        output: String,

        /// Game visitor configuration
        #[clap(flatten)]
        config: VisitorConfig,

        /// Training method to use
        #[command(subcommand)]
        subcommand: TrainMethodCommand,
    },
    /// Starts a process that writes samples to a shared memory file on demand (e.g. for training)
    SamplesService {
        /// List of CSV files to read games.
        /// CSVs must not have headers and rows must have the following format: fen, score, bestmove
        inputs: Vec<String>,

        /// The shared memory file to write the samples. Must have the correct size, otherwise it will panic
        shmem: String,

        /// The feature set to use
        #[arg(long, value_name = "feature-set")]
        #[clap(value_enum)]
        feature_set: FeatureSetChoice,

        /// Batch size
        #[arg(long, default_value = "4096")]
        batch_size: usize,
    },
    /// This command collects information about samples. This is used to create the TopK feature set.
    CollectStats,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Cli::parse();

    match args.command {
        Commands::CollectStats => todo!(),
        Commands::BuildDataset {
            input,
            output,
            config,
            subcommand,
        } => todo!(),
        Commands::SamplesService {
            inputs,
            shmem,
            feature_set,
            batch_size,
        } => todo!(),
    }

    /*

    let count = 0;
    let start = std::time::Instant::now();

    for filename in args.inputs {
        let file = File::open(filename)?;
        let mut buffer = BufReader::new(file);

        let mut fen_bytes = Vec::with_capacity(128);
        let mut score_bytes = Vec::with_capacity(128);
        let mut bestmove_bytes = Vec::with_capacity(128);

        let mut to_move_features = Vec::with_capacity(100_000);
        let mut other_features = Vec::with_capacity(100_000);

        loop {
            fen_bytes.clear();
            score_bytes.clear();
            bestmove_bytes.clear();

            buffer.read_until(b',', &mut fen_bytes).unwrap();
            buffer.read_until(b',', &mut score_bytes).unwrap();
            buffer.read_until(b'\n', &mut bestmove_bytes).unwrap();

            if fen_bytes.is_empty() {
                break;
            }

            // remove trailing comma and newline
            fen_bytes.pop();
            score_bytes.pop();
            bestmove_bytes.pop();

            let fen = Fen::from_ascii(fen_bytes.as_slice())?;
            //let score = parts[1].parse::<f32>()?;
            let bestmove = Uci::from_ascii(bestmove_bytes.as_slice())?;

            let position: Chess = fen.into_position(CastlingMode::Standard)?;
            let board = position.board();

            nn::feature_set::basic::Basic::init(board, position.turn(), &mut to_move_features);
            nn::feature_set::basic::Basic::init(
                board,
                position.turn().other(),
                &mut other_features,
            );

            //count += 1;
            //if count % 100_000 == 0 {
            //    println!("Processed {} positions in {:?}", count, start.elapsed());
            //}
        }
    }

    println!("Processed {} positions", count);

    */

    Ok(())
}
