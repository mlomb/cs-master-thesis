use ort::Session;
use std::{hash::Hash, rc::Rc};
use thesis::{
    algos::alphabeta::alphabeta,
    core::{
        agent::Agent,
        evaluator::PositionEvaluator,
        outcome::Outcome,
        position::{self, Position},
        result::SearchResult,
    },
};

use crate::{
    nn::{NNEvaluator, NNValue},
    nn_agent::NNAgent,
    ringbuffer_set::RingBufferSet,
};

pub struct Trainer<P> {
    win_positions: RingBufferSet<P>,
    loss_positions: RingBufferSet<P>,
}

impl<P> Trainer<P>
where
    P: Position + Eq + Hash,
{
    pub fn new(capacity: usize) -> Self {
        Trainer {
            win_positions: RingBufferSet::new(capacity),
            loss_positions: RingBufferSet::new(capacity),
        }
    }

    pub fn generate_samples(&mut self, session: Rc<NNEvaluator>)
    where
        NNEvaluator: PositionEvaluator<P, NNValue>,
    {
        let mut agent = NNAgent::new(session.clone());
        let mut position = P::initial();
        let mut history = vec![position.clone()];

        while let None = position.status() {
            let chosen_action = agent
                .next_action(&position)
                .expect("agent to return action");

            position = position.apply_action(&chosen_action);
            history.push(position.clone());
        }

        // WLWLWLWL
        //        ↑
        // LWLWLWL
        //       ↑
        assert_eq!(position.status(), Some(Outcome::Loss));

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
            "win: {}, loss: {}",
            self.win_positions.len(),
            self.loss_positions.len()
        );
    }
}
