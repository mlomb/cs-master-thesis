use ort::{Environment, ExecutionProvider, SessionBuilder};
use thesis::{
    games::connect4::Connect4,
    nn::{deep_cmp::DeepCmpTrainer, shmem::open_shmem},
};

fn main() -> ort::Result<()> {
    println!("Hello, world!");

    let environment = Environment::builder()
        .with_execution_providers([ExecutionProvider::CPU(Default::default())])
        .build()?
        .into_arc();

    let session = SessionBuilder::new(&environment)?
        .with_intra_threads(1)?
        .with_model_from_file("models/best/onnx_model.onnx")?;

    let mut trainer = DeepCmpTrainer::<Connect4>::new(10000, session);

    for _i in 0..100 {
        //trainer.generate_samples();
    }

    // Get pointer to the shared memory
    let shmem = open_shmem("deep_cmp_shmem-signal", 8096).unwrap();

    let r = open_shmem("deep_cmp_shmem-inputs", 4096).unwrap();
    let j = open_shmem("deep_cmp_shmem-outputs", 4096).unwrap();

    let raw_ptr = shmem.as_ptr();
    dbg!(shmem.len());
    dbg!(raw_ptr);

    for i in 0..10000 {
        unsafe {
            *raw_ptr = i as u8;
            println!("Wrote {}", i);
        }
        // sleep
        std::thread::sleep(std::time::Duration::from_millis(1000));
    }

    Ok(())
}
