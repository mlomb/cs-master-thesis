use ort::{CPUExecutionProvider, Environment};
use thesis::{
    games::connect4::Connect4,
    nn::{deep_cmp::DeepCmpTrainer, model_management},
};

fn main() -> ort::Result<()> {
    println!("Hello, world!");

    let environment = Environment::builder()
        .with_execution_providers([CPUExecutionProvider::default().build()])
        .with_name("deep_cmp")
        .build()?
        .into_arc();

    let mut trainer = DeepCmpTrainer::<Connect4>::new(10000, 2048, environment);

    for _i in 0..1000 {
        trainer.generate_samples();
        trainer.train();
        trainer.evaluate();
        //std::thread::sleep(std::time::Duration::from_secs(10000));
    }

    model_management::Pepe::new("models").latest();

    Ok(())
}
