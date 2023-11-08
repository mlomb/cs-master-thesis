use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
};

use ndarray::*;
use ort::{Environment, ExecutionProvider, GraphOptimizationLevel, Session, SessionBuilder, Value};
use thesis::{
    algos::{alphabeta::alphabeta, negamax::negamax},
    core::{
        position::Position,
        value::{self, DefaultValuePolicy},
    },
    games::{connect4::Connect4, connect4_strat::Connect4BasicEvaluator},
};

fn main() -> ort::Result<()> {
    tracing_subscriber::fmt::init();

    println!(
        "CUDA available: {}",
        ExecutionProvider::CUDA(Default::default()).is_available()?
    );

    let environment = Environment::builder()
        .with_execution_providers([ExecutionProvider::CPU(Default::default())])
        .build()?
        .into_arc();

    let session = SessionBuilder::new(&environment)?
        .with_intra_threads(1)?
        .with_model_from_file("models/best/onnx_model.onnx")?;

    println!("Model loaded {:?}", session);

    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    struct NNValue {
        pos: Connect4,
        pov: bool, // should flip POV?
    }

    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    struct Pair(NNValue, NNValue);

    struct NNEvaluator;
    struct NNValuePolicy {
        session: Session,
        inferences: usize,
        hs: HashMap<Pair, Ordering>,

        hits: usize,
        misses: usize,
    }

    impl thesis::core::evaluator::PositionEvaluator<Connect4, NNValue> for NNEvaluator {
        fn eval(&self, state: &Connect4) -> NNValue {
            NNValue {
                pos: state.clone(),
                pov: false,
            }
        }
    }

    impl value::ValuePolicy<NNValue> for NNValuePolicy {
        fn compare(&mut self, left: &NNValue, right: &NNValue) -> std::cmp::Ordering {
            let pair = Pair(left.clone(), right.clone());

            if let Some(ordering) = self.hs.get(&pair) {
                self.hits += 1;
                return *ordering;
            }

            self.misses += 1;

            let input = Array5::<f32>::zeros((1, 7, 6, 2, 2));
            let outputs = self.session.run(ort::inputs![input].unwrap()).unwrap();
            let data = outputs[0]
                .extract_tensor::<f32>()
                .unwrap()
                .view()
                .t()
                .into_owned();

            self.inferences += 1;

            let res = if data[[0, 0]] < data[[1, 0]] {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Less
            };

            self.hs.insert(pair, res);
            res
        }

        fn opposite(&mut self, value: &NNValue) -> NNValue {
            NNValue {
                pos: value.pos.clone(),
                pov: !value.pov,
            }
        }
    }

    let mut spec = NNValuePolicy {
        session,
        inferences: 0,
        hs: HashMap::new(),
        hits: 0,
        misses: 0,
    };

    let now = std::time::Instant::now();

    //let (r, a) = minimax(&c4, 5, &mut spec, &NNEvaluator);

    // loop minimax

    let mut position = Connect4::initial();

    loop {
        let (result, best_action) = alphabeta(&position, 6, &mut spec, &NNEvaluator);

        println!("Position:\n{:}", position);
        println!("Result: {:?}", result);
        println!("Best action: {:?}", best_action);
        println!("Inferences: {}", spec.inferences);
        println!("Hits: {} / {}", spec.hits, spec.hits + spec.misses);
        println!("--------------------------");
        spec.inferences = 0;
        /*
        spec.hits = 0;
        spec.misses = 0;
        spec.hs.clear();
        */

        if let Some(action) = best_action {
            position = position.apply_action(&action);
        } else {
            break;
        }
    }

    println!("Elapsed: {:?}", now.elapsed());

    Ok(())
}
