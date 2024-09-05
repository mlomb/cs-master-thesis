use clap::Args;
use nn::{
    feature_set::build::build_feature_set,
    nnue::{accumulator::NnueAccumulator, model::NnueModel},
};
use shakmaty::{fen::Fen, CastlingMode, Chess, Position};
use std::{cell::RefCell, rc::Rc};

#[derive(Args)]
pub struct InfoCommand {
    /// If provided, it will print the number of features of the given feature set
    #[arg(long, value_name = "feature-set")]
    feature_set: Option<String>,

    /// If provided, it will print the features of the given FEN, based on the given feature set
    #[arg(long, value_name = "fen")]
    fen: Option<String>,

    /// If provided, it will print the evaluation of the given FEN using the NNUE model
    /// The feature set given may not coincide.
    #[arg(long, value_name = "nn", requires = "fen")]
    nn: Option<String>,
}

pub fn info(cmd: InfoCommand) {
    if let Some(feature_set) = cmd.feature_set {
        let feature_set = build_feature_set(&feature_set);

        // print number of features
        println!("{}", feature_set.num_features());

        if let Some(ref fen) = cmd.fen {
            let position: Chess = Fen::from_ascii(fen.as_bytes())
                .unwrap()
                .into_position(CastlingMode::Standard)
                .unwrap();

            let mut features = vec![];

            feature_set.active_features(
                position.board(),
                position.turn(),
                position.turn(),
                &mut features,
            );
            features.sort();

            // print pov features
            for x in &features {
                print!("{} ", x);
            }
            println!();

            features.clear();
            feature_set.active_features(
                position.board(),
                position.turn(),
                position.turn().other(),
                &mut features,
            );
            features.sort();

            // print opp features
            for x in &features {
                print!("{} ", x);
            }
            println!();
        }
    }

    if let Some(fen) = cmd.fen {
        let position: Chess = Fen::from_ascii(fen.as_bytes())
            .unwrap()
            .into_position(CastlingMode::Standard)
            .unwrap();

        if let Some(nn_file) = cmd.nn {
            let model = NnueModel::load(&nn_file).unwrap();
            let mut accum = NnueAccumulator::new(Rc::new(RefCell::new(model)));

            accum.refresh(&position, shakmaty::Color::White);
            accum.refresh(&position, shakmaty::Color::Black);

            let eval = accum.forward(position.turn());

            // print evaluation
            println!("{}", eval);
        }
    }
}
