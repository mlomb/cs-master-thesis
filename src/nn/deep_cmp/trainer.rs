use super::encoding::TensorEncodeable;
use super::samples::Samples;
use super::service::DeepCmpService;
use crate::core::agent::Agent;
use crate::core::r#match::play_match;
use crate::core::tournament::Tournament;
use crate::nn::shmem::open_shmem;
use crate::{core::position::Position, nn::deep_cmp::agent::DeepCmpAgent};
use indicatif::{ProgressBar, ProgressStyle};
use ndarray::{ArrayViewMut, IxDyn};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use shared_memory::Shmem;
use std::hash::Hash;
use std::sync::{Arc, Mutex};

pub struct DeepCmpTrainer<P> {
    version: usize,
    batch_size: usize,
    samples: Mutex<Samples<P>>,

    best_model: Arc<DeepCmpService<P>>,

    // shared memory with Python for training
    // we need to keep these alive, since we
    // use them in the tensors below
    status_shmem: Shmem,
    inputs_shmem: Shmem,
    outputs_shmem: Shmem,
}

impl<P> DeepCmpTrainer<P>
where
    P: Position + TensorEncodeable + Eq + Hash + Sync + Send + 'static,
{
    pub fn new(capacity: usize, batch_size: usize) -> Self {
        let inputs_size = batch_size * P::input_shape().iter().product::<usize>();
        let outputs_size = batch_size * P::output_shape().iter().product::<usize>();

        // create shared memory buffers for training
        let status_shmem = open_shmem("deepcmp-status", 2 * 4).unwrap();
        let inputs_shmem = open_shmem("deepcmp-inputs", inputs_size * 4).unwrap();
        let outputs_shmem = open_shmem("deepcmp-outputs", outputs_size * 4).unwrap();

        DeepCmpTrainer {
            version: 1,
            batch_size,
            samples: Mutex::new(Samples::new(capacity)),

            // load best model
            best_model: Arc::new(DeepCmpService::new("models/initial/onnx_model.onnx")),

            // initialize required shared memory buffers for training
            status_shmem,
            inputs_shmem,
            outputs_shmem,
        }
    }

    pub fn generate_samples(&mut self) {
        let n = 50;
        let pb = ProgressBar::new(n).with_style(
            ProgressStyle::with_template(
                "{spinner:.green} {prefix} [Elapsed {elapsed_precise}] (ETA {eta}) [{bar:.cyan/blue}] {human_pos}/{human_len}  {per_sec} ",
            )
            .unwrap()
            .progress_chars("#987654321-")
        ).with_prefix("Generating samples");

        (0..n).into_par_iter().for_each(|_| {
            let mut agent1 = DeepCmpAgent::new(self.best_model.clone(), [8, 8, 8, 4, 3, 2]);
            let mut agent2 = DeepCmpAgent::new(self.best_model.clone(), [8, 8, 8, 4, 3, 2]);
            let mut history = Vec::<P>::new();

            let outcome = play_match(&mut agent1, &mut agent2, Some(&mut history));

            self.samples.lock().unwrap().add_match(&history, outcome);

            pb.inc(1);
        });

        pb.finish();
    }

    /// Prepares and fills the training data
    pub fn train(&mut self) {
        // extract shapes from Position
        let mut inputs_shape_vec = P::input_shape();
        let mut outputs_shape_vec = P::output_shape();

        // add train batch axis
        inputs_shape_vec.insert(0, self.batch_size);
        outputs_shape_vec.insert(0, self.batch_size);

        // build ndarray dyn shapes
        let inputs_shape = IxDyn(&inputs_shape_vec);
        let outputs_shape = IxDyn(&outputs_shape_vec);

        let mut inputs_tensor = unsafe {
            ArrayViewMut::from_shape_ptr(inputs_shape, self.inputs_shmem.as_ptr() as *mut f32)
        };
        let mut outputs_tensor = unsafe {
            ArrayViewMut::from_shape_ptr(outputs_shape, self.outputs_shmem.as_ptr() as *mut f32)
        };

        self.samples.lock().unwrap().write_samples(
            self.batch_size,
            &mut inputs_tensor,
            &mut outputs_tensor,
        );

        let status_ptr = self.status_shmem.as_ptr() as *mut u32;

        unsafe {
            // write version
            status_ptr.offset(1).write(self.version as u32);
            // write batch size
            status_ptr.offset(2).write(self.batch_size as u32);

            // mark as ready so Python can start training
            status_ptr.offset(0).write(1);
        };

        // wait for Python to finish training
        while unsafe { status_ptr.offset(0).read() } == 1 {
            std::thread::sleep(std::time::Duration::from_millis(100));
        }

        self.version += 1;
    }

    pub fn evaluate(&mut self) {
        let best_model = self.best_model.clone();
        let candidate_model = Arc::new(DeepCmpService::new("models/candidate/onnx_model.onnx"));
        let cloned_candidate_model = candidate_model.clone();

        let best_closure =
            move || Box::new(DeepCmpAgent::new(best_model.clone(), [7, 7, 7])) as Box<dyn Agent<_>>;
        let candidate_closure = move || {
            Box::new(DeepCmpAgent::new(candidate_model.clone(), [7, 7, 7])) as Box<dyn Agent<_>>
        };

        let res = Tournament::new()
            //.add_agent("random", &|| Box::new(RandomAgent {}))
            .add_agent("best", &best_closure)
            .add_agent("candidate", &candidate_closure)
            .num_matches(100)
            .show_progress(true)
            .use_parallel(true)
            .run();

        println!("{}", res);

        if res.win_rate("candidate") > 0.55 {
            println!("Candidate is better, replacing best model");
            self.best_model = cloned_candidate_model;
        } else {
            println!("Candidate is not better, keeping best model");
        }
    }
}
