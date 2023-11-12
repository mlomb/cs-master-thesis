use ort::{Environment, ExecutionProvider, SessionBuilder};
use thesis::{
    games::connect4::Connect4,
    nn::{deep_cmp::DeepCmpTrainer, model_management},
};

fn main() -> ort::Result<()> {
    println!("Hello, world!");

    let environment = Environment::builder()
        .with_execution_providers([ExecutionProvider::CPU(Default::default())])
        .with_name("deep_cmp")
        .build()?
        .into_arc();

    let session = SessionBuilder::new(&environment)?
        .with_intra_threads(1)?
        .with_model_from_file("models/best/onnx_model.onnx")?;

    let mut trainer = DeepCmpTrainer::<Connect4>::new(10000, 600, session);

    for _i in 0..5 {
        trainer.generate_samples();
        trainer.train();
    }

    model_management::Pepe::new("models").latest();

    Ok(())
}
