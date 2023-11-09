mod cmp;
mod nn;
mod nn_encoding;
mod ringbuffer_set;
mod trainer;

use crate::{
    nn::{NNEvaluator, NNValuePolicy},
    trainer::Trainer,
};
use ort::{Environment, ExecutionProvider, SessionBuilder};
use thesis::games::connect4::Connect4;

fn main() -> ort::Result<()> {
    println!("Hello, world!");

    let environment = Environment::builder()
        .with_execution_providers([ExecutionProvider::CPU(Default::default())])
        .build()?
        .into_arc();

    let session = SessionBuilder::new(&environment)?
        .with_intra_threads(1)?
        .with_model_from_file("models/best/onnx_model.onnx")?;

    let mut spec = NNValuePolicy::new(session);

    let mut trainer = Trainer::<Connect4>::new(10000);

    loop {
        trainer.generate_samples(&mut spec, &NNEvaluator);
    }
}
