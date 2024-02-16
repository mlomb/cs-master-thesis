use clap::Subcommand;

#[derive(Subcommand)]
pub enum TrainMethodCommand {
    /// Generates a dataset with three columns: FEN strings of the positions P, Q and R.
    /// Given a transition P → Q in a game, R is selected from a legal move from P while R != Q.
    PQR,
    /// Generates a dataset with three columns: FEN string of a position, its score and the best move (both given by the engine)
    Eval {
        /// UCI engine command to use for evaluation
        #[arg(long, value_name = "engine")]
        engine: String,

        /// Target depth for search
        #[arg(long, value_name = "depth", default_value = "13")]
        depth: usize,
    },
}
