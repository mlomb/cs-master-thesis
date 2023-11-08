use std::hash::Hash;
use thesis::{
    algos::alphabeta::alphabeta,
    core::{
        evaluator::PositionEvaluator,
        outcome::Outcome,
        position::{self, Position},
    },
};

use crate::{
    nn::{NNEvaluator, NNValue, NNValuePolicy},
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

    pub fn generate_samples(&mut self, spec: &mut NNValuePolicy, evaluator: &NNEvaluator)
    where
        NNEvaluator: PositionEvaluator<P, NNValue>,
    {
        let mut history = Vec::new();

        let mut position = P::initial();
        loop {
            history.push(position.clone());

            let (_, best_action) = alphabeta(&position, 3, spec, evaluator);

            if let Some(action) = best_action {
                position = position.apply_action(&action);
            } else {
                break;
            }
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
    }
}
