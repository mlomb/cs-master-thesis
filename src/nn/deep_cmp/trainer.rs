use super::evaluator::DeepCmpEvaluator;
use super::ringbuffer_set::RingBufferSet;
use crate::{
    algos::alphabeta::alphabeta,
    core::{
        agent::Agent,
        evaluator::PositionEvaluator,
        outcome::Outcome,
        position::{self, Position},
        result::SearchResult,
    },
    nn::{deep_cmp::agent::DeepCmpAgent, nn_encoding::TensorEncodeable},
};
use ort::Session;
use std::{collections::HashSet, hash::Hash, rc::Rc};

pub struct DeepCmpTrainer<P> {
    win_positions: RingBufferSet<P>,
    loss_positions: RingBufferSet<P>,
    all_positions: HashSet<P>,
    evaluator: Rc<DeepCmpEvaluator<P>>,
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
            evaluator: Rc::new(DeepCmpEvaluator::new(session)),
        }
    }

    pub fn generate_samples(&mut self) {
        let mut agent = DeepCmpAgent::new(self.evaluator.clone());
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
}
