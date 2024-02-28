use crate::method::ReadSample;
use clap::{Args, Subcommand, ValueEnum};
use nn::feature_set::{basic::Basic, FeatureSet};
use shared_memory::ShmemConf;
use std::{
    error::Error,
    fs::File,
    io::{self, BufRead, BufReader, Cursor, Read, Seek, Write},
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
    let mut shmem = ShmemConf::new().os_id(cmd.shmem).open()?;

    // initialize feature set
    let feature_set: Box<dyn FeatureSet> = match cmd.feature_set {
        FeatureSetChoice::Basic => Box::new(Basic::new()),
        FeatureSetChoice::HalfKP => todo!(),
        FeatureSetChoice::TopK20 => todo!(),
    };

    let mut method = match cmd.subcommand {
        MethodSubcommand::PQR => PQR::new(),
    };

    let expected_size = method.sample_size(&feature_set) * cmd.batch_size;

    assert_eq!(shmem.len(), expected_size);
    let mut cursor = Cursor::new(unsafe { shmem.as_slice_mut() });

    //let mut a = vec![0u8; expected_size];
    //let mut cursor = Cursor::new(a.as_mut_slice());

    let mut in_batch = 0;

    // loop over the dataset indefinitely
    loop {
        // loop over every input file
        for filename in &cmd.inputs {
            let file = File::open(filename)?;

            // decompress if necessary
            let reader: Box<dyn io::Read> = if filename.ends_with(".zst") {
                Box::new(zstd::Decoder::new(file)?)
            } else {
                Box::new(file)
            };

            let mut reader = BufReader::with_capacity(32 * 8192, reader);

            // loop for every sample in file
            while reader.has_data_left()? {
                // write sample
                method.read_sample(&mut reader, &mut cursor, &feature_set);
                in_batch += 1;

                if in_batch == cmd.batch_size {
                    // we have filled the current batch with samples
                    // send a signal to the consumer (1 byte)
                    io::stdout().write_all(&[64])?;
                    io::stdout().flush()?;

                    // now we wait for the consumer to signal that it has finished copying the data (1 byte)
                    io::stdin().read_exact(&mut [0])?;

                    // reset cursor to the beginning
                    cursor.rewind()?;
                    in_batch = 0;
                }
            }
        }
    }
}
