#![feature(bufread_skip_until)]
#![feature(slice_pattern)]

mod batch_loader;
mod convert;
mod encode;
mod feature_set_size;
mod format;
mod method;

use crate::batch_loader::batch_loader;
use crate::convert::convert;
use crate::feature_set_size::feature_set_size;
use batch_loader::BatchLoaderCommand;
use clap::{Parser, Subcommand};
use convert::ConvertCommand;
use feature_set_size::FeatureSetSizeCommand;
use std::error::Error;

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Convert from .plain to .plainpack
    Convert(ConvertCommand),
    /// Starts a process that writes samples to a shared memory file on demand (e.g. for training)
    BatchLoader(BatchLoaderCommand),
    /// Displays the size of a feature set
    FeatureSetSize(FeatureSetSizeCommand),
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Cli::parse();

    match args.command {
        Commands::Convert(cmd) => convert(cmd),
        Commands::BatchLoader(cmd) => batch_loader(cmd),
        Commands::FeatureSetSize(cmd) => feature_set_size(cmd),
    }
}
