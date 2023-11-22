use ndarray::*;
use ort::{CPUExecutionProvider, Session};

fn main() -> ort::Result<()> {
    tracing_subscriber::fmt::init();

    ort::init()
        .with_execution_providers([CPUExecutionProvider::default().build()])
        .commit()?;

    let session = Session::builder()?
        .with_intra_threads(1)?
        .with_model_from_file("models/best/onnx_model.onnx")?;

    println!("Model loaded {:?}", session);

    {
        for _ in 0..5 {
            let now = std::time::Instant::now();
            for _ in 0..1000 {
                let input = Array4::<f32>::zeros((100, 7, 6, 4));
                let outputs = session.run(ort::inputs![input]?)?;
                let _data = outputs[0].extract_tensor::<f32>()?.view().t().into_owned();
            }
            println!("Elapsed: {:?}", now.elapsed());
        }
    }

    {
        for _ in 0..5 {
            let now = std::time::Instant::now();
            for _ in 0..100000 {
                let input = Array4::<f32>::zeros((1, 7, 6, 4));
                let outputs = session.run(ort::inputs![input]?)?;
                let _data = outputs[0].extract_tensor::<f32>()?.view().t().into_owned();
            }
            println!("Elapsed: {:?}", now.elapsed());
        }
    }

    Ok(())
}
