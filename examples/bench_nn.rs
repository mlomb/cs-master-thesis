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
        .with_intra_threads(1)?
        .with_model_from_file("models/best/onnx_model.onnx")?;

    println!("Model loaded {:?}", session);

    {
        for _ in 0..5 {
            let now = std::time::Instant::now();
            for _ in 0..1000 {
                let input = Array5::<f32>::zeros((100, 7, 6, 2, 2));
                let outputs = session.run(ort::inputs![input]?)?;
                let data = outputs[0].extract_tensor::<f32>()?.view().t().into_owned();
            }
            println!("Elapsed: {:?}", now.elapsed());
        }
    }

    {
        for _ in 0..5 {
            let now = std::time::Instant::now();
            for _ in 0..100000 {
                let input = Array5::<f32>::zeros((1, 7, 6, 2, 2));
                let outputs = session.run(ort::inputs![input]?)?;
                let data = outputs[0].extract_tensor::<f32>()?.view().t().into_owned();
            }
            println!("Elapsed: {:?}", now.elapsed());
        }
    }

    Ok(())
}
