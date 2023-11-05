use ndarray::*;
use ort::{Environment, ExecutionProvider, GraphOptimizationLevel, SessionBuilder, Value};

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

    {
        for _ in 0..5 {
            let now = std::time::Instant::now();
            for _ in 0..1000 {
                let input = Array5::<f32>::zeros((100, 7, 6, 2, 2));
                let output = Array2::<f32>::zeros((100, 2));

                let in_value = Value::from_array(input)?;
                let out_value = Value::from_array(output)?;

                let mut binding = session.create_binding()?;
                binding.bind_input("input_1", in_value)?;
                binding.bind_output("dense_4", out_value)?;
                binding.run()?;
            }
            println!("Elapsed: {:?}", now.elapsed());
        }
    }

    {
        for _ in 0..5 {
            let now = std::time::Instant::now();
            for _ in 0..100000 {
                let input = Array5::<f32>::zeros((1, 7, 6, 2, 2));
                let output = Array2::<f32>::zeros((1, 2));

                let in_value = Value::from_array(input)?;
                let out_value = Value::from_array(output)?;

                let mut binding = session.create_binding()?;
                binding.bind_input("input_1", in_value)?;
                binding.bind_output("dense_4", out_value)?;
                binding.run()?;
            }
            println!("Elapsed: {:?}", now.elapsed());
        }
    }

    Ok(())
}
