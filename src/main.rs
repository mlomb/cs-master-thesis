use ort::CPUExecutionProvider;
use thesis::{games::connect4::Connect4, nn::deep_cmp::DeepCmpTrainer};

fn main() -> ort::Result<()> {
    println!("Hello, world!");

    ort::init()
        .with_execution_providers([CPUExecutionProvider::default().build()])
        .with_name("deep_cmp")
        .commit()?;

    let mut trainer = DeepCmpTrainer::<Connect4>::new(5000, 1024);

    for _i in 0..100000 {
        trainer.generate_samples();
        trainer.train();
        //trainer.evaluate();
        std::thread::sleep(std::time::Duration::from_secs(1));
    }

    // model_management::Pepe::new("models").latest();

    Ok(())
}
