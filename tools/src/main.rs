#![feature(buf_read_has_data_left)]

mod batch_loader;
mod encode;
mod feature_set_size;
mod method;

use crate::batch_loader::batch_loader;
use crate::feature_set_size::feature_set_size;
use batch_loader::BatchLoaderCommand;
use clap::{Parser, Subcommand};
use feature_set_size::FeatureSetSizeCommand;
use std::error::Error;

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Starts a process that writes samples to a shared memory file on demand (e.g. for training)
    BatchLoader(BatchLoaderCommand),
    /// Displays the size of a feature set
    FeatureSetSize(FeatureSetSizeCommand),
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Cli::parse();

    match args.command {
        Commands::BatchLoader(cmd) => batch_loader(cmd),
        Commands::FeatureSetSize(cmd) => feature_set_size(cmd),
    }
}
