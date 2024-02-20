mod build_dataset;
mod game_visitor;
mod method;
mod samples_service;
mod uci_engine;

use crate::build_dataset::build_dataset;
use crate::samples_service::samples_service;
use build_dataset::BuildDatasetCommand;
use clap::{Parser, Subcommand};
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
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Cli::parse();

    match args.command {
        Commands::BuildDataset(cmd) => build_dataset(cmd),
        Commands::SamplesService(cmd) => samples_service(cmd),
    }
}
