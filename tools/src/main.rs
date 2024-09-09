#![feature(bufread_skip_until)]
#![feature(slice_pattern)]
#![feature(iter_map_windows)]

mod batch_loader;
mod convert;
mod info;
mod method;
mod plain_format;
mod pos_encoding;
mod stats;

use crate::batch_loader::batch_loader;
use crate::convert::convert;
use crate::info::info;
use crate::stats::stats;
use batch_loader::BatchLoaderCommand;
use clap::{Parser, Subcommand};
use convert::ConvertCommand;
use info::InfoCommand;
use stats::StatsCommand;
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
    /// Prints information: feature set size, one hot encoding of a position and the evaluation
    Info(InfoCommand),
    /// Gather stats on a dataset
    Stats(StatsCommand),
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Cli::parse();

    match args.command {
        Commands::Convert(cmd) => convert(cmd),
        Commands::BatchLoader(cmd) => batch_loader(cmd),
        Commands::Info(cmd) => Ok(info(cmd)),
        Commands::Stats(cmd) => Ok(stats(cmd)),
    }
}
