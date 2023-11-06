use ndarray::*;
use ort::{Environment, ExecutionProvider, GraphOptimizationLevel, Session, SessionBuilder, Value};
use thesis::{
    algos::minimax::minimax,
    core::{position::Position, value},
    games::connect4::Connect4,
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
        .with_optimization_level(GraphOptimizationLevel::Level1)?
        .with_intra_threads(1)?
        .with_model_from_file("models/best/onnx_model.onnx")?;

    println!("Model loaded {:?}", session);

    let c4 = Connect4::initial();

    #[derive(Debug)]
    struct NNValue {
        pos: Connect4,
        pov: bool, // should flip POV?
    }
    struct NNEvaluator;
    struct NNValuePolicy {
        session: Session,
        inferences: usize,
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
            let input = Array5::<f32>::zeros((1, 7, 6, 2, 2));
            let output = Array2::<f32>::zeros((1, 2));

            let in_value = Value::from_array(input).unwrap();
            let out_value = Value::from_array(output).unwrap();

            let mut binding = self.session.create_binding().unwrap();
            binding.bind_input("input", in_value).unwrap();
            binding.bind_output("output", out_value).unwrap();
            let outputs = binding.run().unwrap();

            let data = outputs[0]
                .extract_tensor::<f32>()
                .unwrap()
                .view()
                .t()
                .into_owned();

            self.inferences += 1;

            if data[[0, 0]] > data[[1, 0]] {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Less
            }
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
    };

    let now = std::time::Instant::now();

    let (r, a) = minimax(&c4, 5, &mut spec, &NNEvaluator);

    println!("Result: {:?}", r);
    println!("Action: {:?}", a);

    println!("Elapsed: {:?}", now.elapsed());

    println!("Inferences: {}", spec.inferences);

    Ok(())
}
