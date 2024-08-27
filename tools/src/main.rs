#![feature(buf_read_has_data_left)]

mod encode;
mod feature_set_size;
mod method;
mod samples_service;

use crate::feature_set_size::feature_set_size;
use crate::samples_service::samples_service;
use clap::{Parser, Subcommand};
use feature_set_size::FeatureSetSizeCommand;
use samples_service::SamplesServiceCommand;
use std::error::Error;

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Starts a process that writes samples to a shared memory file on demand (e.g. for training)
    SamplesService(SamplesServiceCommand),
    /// Displays the size of a feature set
    FeatureSetSize(FeatureSetSizeCommand),
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Cli::parse();

    match args.command {
        Commands::SamplesService(cmd) => samples_service(cmd),
        Commands::FeatureSetSize(cmd) => feature_set_size(cmd),
    }
}
