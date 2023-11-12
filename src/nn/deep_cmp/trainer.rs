use super::service::DeepCmpService;
use super::{encoding::TensorEncodeable, ringbuffer_set::RingBufferSet};
use crate::{
    core::{agent::Agent, outcome::Outcome, position::Position},
    nn::{deep_cmp::agent::DeepCmpAgent, shmem::open_shmem},
};
use ort::Session;
use shared_memory::Shmem;
use std::{cell::RefCell, collections::HashSet, hash::Hash, rc::Rc};

pub struct DeepCmpTrainer<P> {
    win_positions: RingBufferSet<P>,
    loss_positions: RingBufferSet<P>,
    all_positions: HashSet<P>,
    service: Rc<RefCell<DeepCmpService<P>>>,

    // shared memory with Python for training
    signal_shmem: Shmem,
    inputs_shmem: Shmem,
    outputs_shmem: Shmem,
}

impl<P> DeepCmpTrainer<P>
where
    P: Position + TensorEncodeable + Eq + Hash,
{
    pub fn new(capacity: usize, session: Session) -> Self {
        DeepCmpTrainer {
            win_positions: RingBufferSet::new(capacity),
            loss_positions: RingBufferSet::new(capacity),
            all_positions: HashSet::new(),
            service: Rc::new(RefCell::new(DeepCmpService::new(session))),

            // initialize required shared memory buffers for training
            signal_shmem: open_shmem("deep_cmp_shmem-signal", 4096).unwrap(),
            inputs_shmem: open_shmem("deep_cmp_shmem-inputs", 4096).unwrap(),
            outputs_shmem: open_shmem("deep_cmp_shmem-outputs", 4096).unwrap(),
        }
    }

    pub fn generate_samples(&mut self) {
        let mut agent = DeepCmpAgent::new(self.service.clone());
        let mut position = P::initial();
        let mut history = vec![position.clone()];

        while let None = position.status() {
            let chosen_action = agent
                .next_action(&position)
                .expect("agent to return action");

            position = position.apply_action(&chosen_action);
            history.push(position.clone());
            self.all_positions.insert(position.clone());
        }

        let status = position.status();

        // ignore draws
        if let Some(Outcome::Draw) = status {
            return;
        }

        // We expect a loss, since the POV is changed after the last move
        // WLWLWLWL
        //        ↑
        // LWLWLWL
        //       ↑
        assert_eq!(status, Some(Outcome::Loss));

        // iterate over history in reverse
        // knowing that the last state is a loss
        let mut it = history.iter().rev().peekable();
        while it.peek().is_some() {
            if let Some(pos) = it.next() {
                self.loss_positions.insert(pos.clone());
            }
            if let Some(pos) = it.next() {
                self.win_positions.insert(pos.clone());
            }
        }

        println!(
            "win: {}, loss: {} all: {}",
            self.win_positions.len(),
            self.loss_positions.len(),
            self.all_positions.len()
        );
    }

    /// Prepares and fills the training data
    pub fn train(&mut self) {

        // samples will be tagged like this
        // [W, L]: [1, 0]
        // [L, W]: [0, 1]

        // prepare inputs
    }
}
