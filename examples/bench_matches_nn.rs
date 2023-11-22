use indicatif::{ProgressBar, ProgressStyle};
use ndarray::ArrayD;
use ort::{CPUExecutionProvider, Session};
use ort_batcher::batcher::Batcher;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use std::{sync::Arc, time::Duration};
use thesis::{
    core::{agent::Agent, position::Position, r#match::play_match},
    games::connect4::Connect4,
};

fn main() -> ort::Result<()> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(1024)
        .build_global()
        .unwrap();

    ort::init()
        .with_execution_providers([CPUExecutionProvider::default().build()])
        //.with_execution_providers([CUDAExecutionProvider::default().build()])
        .commit()?;

    let session = Session::builder()?
        .with_intra_threads(4)?
        .with_model_from_file("models/best/onnx_model.onnx")?;

    println!("Model loaded {:?}", session);
    let batcher = Batcher::spawn(session, 512, Duration::from_millis(3));
    let batcher = Arc::new(batcher);

    #[derive(Clone)]
    struct RandomAgent {
        sum: f32,
        //session: Arc<Session>,
        batcher: Arc<Batcher>,
    }
    // unsafe impl Send for RandomAgent {}
    // unsafe impl Sync for RandomAgent {}
    impl Agent<Connect4> for RandomAgent {
        fn next_action(&mut self, position: &Connect4) -> Option<usize> {
            /*
            let input = ArrayD::<f32>::zeros(vec![7, 6, 4]).insert_axis(Axis(0));
            let value = Value::from_array(input.clone()).unwrap();
            self.session.run([value]).unwrap()[0]
                .extract_tensor::<f32>()
                .unwrap()
                .view()
                .index_axis(Axis(0), 0)
                .to_owned();
            */

            let res = self.batcher.run(vec![ArrayD::<f32>::zeros(vec![7, 6, 4])]);
            self.sum += res.unwrap()[0][0];
            position
                .valid_actions()
                .choose(&mut rand::thread_rng())
                .cloned()
        }
    }

    let n = 1 * 1000000;
    let agent = RandomAgent {
        sum: 0.0,
        //session: Arc::new(session),
        batcher: batcher.clone(),
    };

    {
        let start = std::time::Instant::now();
        let pb = ProgressBar::new(n);
        pb.set_style(
            ProgressStyle::with_template(
                "{spinner:.green} [Elapsed {elapsed_precise}] (ETA {eta}) [{bar:.cyan/blue}] {human_pos}/{human_len}  {per_sec} ",
            )
            .unwrap()
            .progress_chars("#987654321-")
        );

        (0..n).into_par_iter().for_each(|_| {
            play_match(&mut agent.clone(), &mut agent.clone(), None);
            pb.inc(1);
        });

        pb.finish();
        println!("elapsed: {:?}", start.elapsed());
    }

    Ok(())
}
