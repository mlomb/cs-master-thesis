#![feature(bufread_skip_until)]
#![feature(slice_pattern)]

mod batch_loader;
mod convert;
mod feature_set_size;
mod method;
mod plain_format;
mod pos_encoding;

use crate::batch_loader::batch_loader;
use crate::convert::convert;
use crate::feature_set_size::feature_set_size;
use batch_loader::BatchLoaderCommand;
use clap::{Parser, Subcommand};
use convert::ConvertCommand;
use feature_set_size::FeatureSetSizeCommand;
use std::error::Error;

/// Collection of tools to enable training of NNUE models
#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Convert from .plain to .plain. This is used to compact plain files
    Convert(ConvertCommand),
    /// Starts a process that writes samples to a shared memory file on demand (e.g. for training)
    BatchLoader(BatchLoaderCommand),
    /// Prints the size of a feature set. Convenient for Python scripts
    FeatureSetSize(FeatureSetSizeCommand),
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Cli::parse();

    match args.command {
        Commands::Convert(cmd) => convert(cmd),
        Commands::BatchLoader(cmd) => batch_loader(cmd),
        Commands::FeatureSetSize(cmd) => Ok(feature_set_size(cmd)),
    }
}
