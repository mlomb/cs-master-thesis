use crate::method::ReadSample;
use clap::{Args, Subcommand, ValueEnum};
use nn::feature_set::{basic::Basic, FeatureSet};
use shared_memory::ShmemConf;
use std::{
    error::Error,
    fs::File,
    io::{BufRead, BufReader},
};

use crate::method::pqr::PQR;

#[derive(ValueEnum, Clone)]
enum FeatureSetChoice {
    Basic,
    HalfKP,
    TopK20,
}

#[derive(Args)]
pub struct SamplesServiceCommand {
    /// List of CSV files to read games.
    /// CSVs must be generated using the `build-dataset` command
    #[arg(long, value_name = "input", required = true)]
    inputs: Vec<String>,

    /// The shared memory file to write the samples. Must have the correct size, otherwise it will panic
    #[arg(long, value_name = "shmem")]
    shmem: String,

    /// The feature set to use
    #[arg(long, value_name = "feature-set")]
    #[clap(value_enum)]
    feature_set: FeatureSetChoice,

    /// Batch size
    #[arg(long, default_value = "4096")]
    batch_size: usize,

    /// Method to use
    #[command(subcommand)]
    subcommand: MethodSubcommand,
}

#[derive(Subcommand)]
pub enum MethodSubcommand {
    PQR,
}

pub fn samples_service(cmd: SamplesServiceCommand) -> Result<(), Box<dyn Error>> {
    // open shared memory file
    //let shmem = ShmemConf::new().flink(cmd.shmem).open()?;

    // initialize feature set
    let feature_set = match cmd.feature_set {
        FeatureSetChoice::Basic => Basic::new(),
        FeatureSetChoice::HalfKP => todo!(),
        FeatureSetChoice::TopK20 => todo!(),
    };

    let mut method = match cmd.subcommand {
        MethodSubcommand::PQR => PQR::new(),
    };

    let buffer_size = feature_set.num_features().div_ceil(64) * cmd.batch_size;
    // assert_eq!(shmem.len(), buffer_size);
    let mut buffer = [0u64; 1 * 41000]; // TODO: put this on the heap

    for filename in cmd.inputs {
        let file = File::open(filename)?;
        let mut file_buffer = BufReader::new(file);

        while file_buffer.has_data_left()? {
            method.read_sample(&mut file_buffer, &mut buffer, &feature_set);
        }
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
    todo!()
}
