use super::service::DeepCmpService;
use super::{encoding::TensorEncodeable, ringbuffer_set::RingBufferSet};
use crate::core::r#match::{play_match, MatchOutcome};
use crate::{
    core::{agent::Agent, outcome::Outcome, position::Position},
    nn::{deep_cmp::agent::DeepCmpAgent, shmem::open_shmem},
};
use ndarray::{ArrayViewMut, ArrayViewMutD, Axis, Dimension, IxDyn};
use ort::Session;
use rand::Rng;
use shared_memory::Shmem;
use std::cmp::Ordering;
use std::{cell::RefCell, collections::HashSet, hash::Hash, rc::Rc};

pub struct DeepCmpTrainer<'a, P> {
    win_positions: RingBufferSet<P>,
    loss_positions: RingBufferSet<P>,
    all_positions: HashSet<P>,
    service: Rc<RefCell<DeepCmpService<P>>>,

    version: usize,
    batch_size: usize,

    // shared memory with Python for training
    // we need to keep these alive, since we
    // use them in the tensors below
    #[allow(dead_code)]
    status_shmem: Shmem,
    #[allow(dead_code)]
    inputs_shmem: Shmem,
    #[allow(dead_code)]
    outputs_shmem: Shmem,

    // tensors (that use the shared memory)
    status_tensor: ArrayViewMutD<'a, u32>,
    inputs_tensor: ArrayViewMutD<'a, f32>,
    outputs_tensor: ArrayViewMutD<'a, f32>,
}

impl<P> DeepCmpTrainer<'_, P>
where
    P: Position + TensorEncodeable + Eq + Hash + 'static,
{
    pub fn new(capacity: usize, batch_size: usize, session: Session) -> Self {
        // extract shapes from Position
        let mut inputs_shape_vec = P::input_shape();
        let mut outputs_shape_vec = P::output_shape();

        // add train batch axis
        inputs_shape_vec.insert(0, batch_size);
        outputs_shape_vec.insert(0, batch_size);

        // build ndarray dyn shapes
        let inputs_shape = IxDyn(&inputs_shape_vec);
        let outputs_shape = IxDyn(&outputs_shape_vec);

        // create shared memory buffers for training
        let status_shmem = open_shmem("deepcmp-status", 2 * 4).unwrap();
        let inputs_shmem = open_shmem("deepcmp-inputs", inputs_shape.size() * 4).unwrap();
        let outputs_shmem = open_shmem("deepcmp-outputs", outputs_shape.size() * 4).unwrap();

        let status_tensor =
            unsafe { ArrayViewMut::from_shape_ptr(IxDyn(&[2]), status_shmem.as_ptr() as *mut u32) };
        let inputs_tensor = unsafe {
            ArrayViewMut::from_shape_ptr(inputs_shape, inputs_shmem.as_ptr() as *mut f32)
        };
        let outputs_tensor = unsafe {
            ArrayViewMut::from_shape_ptr(outputs_shape, outputs_shmem.as_ptr() as *mut f32)
        };

        DeepCmpTrainer {
            win_positions: RingBufferSet::new(capacity),
            loss_positions: RingBufferSet::new(capacity),
            all_positions: HashSet::new(),
            service: Rc::new(RefCell::new(DeepCmpService::new(session))),

            version: 1,
            batch_size,

            // initialize required shared memory buffers for training
            status_shmem,
            inputs_shmem,
            outputs_shmem,

            status_tensor,
            inputs_tensor,
            outputs_tensor,
        }
    }

    pub fn generate_samples(&mut self) {
        let agent = Rc::new(RefCell::new(DeepCmpAgent::new(self.service.clone())));
        let mut history = Vec::<P>::new();

        let outcome = play_match(agent.clone(), agent, Some(&mut history));

        // even positions
        history
            .iter()
            .step_by(2)
            .cloned()
            .for_each(|pos| match outcome {
                MatchOutcome::WinAgent1 => self.win_positions.insert(pos),
                MatchOutcome::WinAgent2 => self.loss_positions.insert(pos),
                _ => (),
            });

        // odd positions
        history
            .iter()
            .skip(1)
            .step_by(2)
            .cloned()
            .for_each(|pos| match outcome {
                MatchOutcome::WinAgent1 => self.loss_positions.insert(pos),
                MatchOutcome::WinAgent2 => self.win_positions.insert(pos),
                _ => (),
            });

        // add all positions to the set
        self.all_positions.extend(history.iter().cloned());

        println!(
            "win: {}, loss: {} all: {}",
            self.win_positions.len(),
            self.loss_positions.len(),
            self.all_positions.len()
        );
    }

    /// Prepares and fills the training data
    pub fn train(&mut self) {
        // clear tensors
        self.status_tensor.fill(0);
        self.inputs_tensor.fill(0.0);
        self.outputs_tensor.fill(0.0);

        let mut rng = rand::thread_rng();

        for i in 0..self.batch_size {
            let win_pos = self.win_positions.sample_one(&mut rng).unwrap();
            let loss_pos = self.loss_positions.sample_one(&mut rng).unwrap();

            let (left_board, ordering, right_board) = if rng.gen_bool(0.5) {
                (win_pos, Ordering::Greater, loss_pos)
            } else {
                (loss_pos, Ordering::Less, win_pos)
            };

            P::encode_input(
                left_board,
                right_board,
                &mut self.inputs_tensor.index_axis_mut(Axis(0), i),
            );
            P::encode_output(
                ordering,
                &mut self.outputs_tensor.index_axis_mut(Axis(0), i),
            );
        }

        let status_ptr = self.status_shmem.as_ptr() as *mut u32;

        unsafe {
            // write version
            status_ptr.offset(1).write(self.version as u32);
            // mark as ready so Python can start training
            status_ptr.offset(0).write(1);
        };

        // wait for Python to finish training
        while unsafe { status_ptr.offset(0).read() } == 1 {
            std::thread::sleep(std::time::Duration::from_millis(100));
        }

        self.version += 1;
    }
}
