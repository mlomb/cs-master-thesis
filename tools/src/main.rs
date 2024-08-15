#![feature(buf_read_has_data_left)]

mod build_dataset;
mod feature_set_size;
mod game_visitor;
mod method;
mod samples_service;
mod uci_engine;

use crate::build_dataset::build_dataset;
use crate::feature_set_size::feature_set_size;
use crate::samples_service::samples_service;
use build_dataset::BuildDatasetCommand;
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
    /// Builds a dataset for a speific training method using PGN files as input. Only one sample per game is extracted.
    BuildDataset(BuildDatasetCommand),
    /// Starts a process that writes samples to a shared memory file on demand (e.g. for training)
    SamplesService(SamplesServiceCommand),
    /// Returns the size of a feature set
    FeatureSetSize(FeatureSetSizeCommand),
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Cli::parse();

    match args.command {
        Commands::BuildDataset(cmd) => build_dataset(cmd),
        Commands::SamplesService(cmd) => samples_service(cmd),
        Commands::FeatureSetSize(cmd) => feature_set_size(cmd),
    }
}
