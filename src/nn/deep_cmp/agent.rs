use super::{
    evaluator::DeepCmpEvaluator, round_robin_sort::rr_sort, service::DeepCmpService,
    value::DeepCmpValue,
};
use crate::{
    algos::alphabeta::alphabeta,
    core::{agent::Agent, position, value::Value},
    nn::nn_encoding::TensorEncodeable,
};
use rand::seq::SliceRandom;
use std::{cell::RefCell, collections::VecDeque, rc::Rc};

/// The agent for the DeepCompare algorithm
///
/// It will use `alphabeta` up to a fixed depth to evaluate all possible actions;
/// then choose a random action from the top-k actions given by the wiggle.
/// The wiggle is a queue of integers that will be used as the k in the top-k actions.
/// If the wiggle is empty it will default to 1, i.e. it will choose the best action.
pub struct DeepCmpAgent<Position> {
    /// Reference back to the service
    service: Rc<RefCell<DeepCmpService<Position>>>,
    /// The depth to use for the search
    target_depth: usize,
    /// Queue of integers to use as the k in the top-k actions, consumed each turn
    random_wiggle: VecDeque<usize>,
}

impl<Position> DeepCmpAgent<Position> {
    pub fn new(service: Rc<RefCell<DeepCmpService<Position>>>) -> Self {
        DeepCmpAgent {
            service,
            target_depth: 3,
            random_wiggle: VecDeque::from([8, 8, 8, 4, 3, 2]),
        }
    }
}

impl<Position> Agent<Position> for DeepCmpAgent<Position>
where
    Position: position::Position + TensorEncodeable,
    DeepCmpValue<Position>: crate::core::value::Value,
{
    fn next_action(&mut self, position: &Position) -> Option<Position::Action> {
        let mut evaluator = DeepCmpEvaluator::new(self.service.clone());
        let actions = position.valid_actions();

        // we will sample an action from the top-k actions
        // where k is the first element of the random_wiggle
        let k = std::cmp::min(self.random_wiggle.pop_front().unwrap_or(1), actions.len());

        let mut results = Vec::new();

        for action in actions {
            let (value, _) = alphabeta(
                &position.apply_action(&action),
                self.target_depth,
                &mut evaluator,
            );

            results.push((value, action));
        }

        // use an special "sorting" algorithm for non-transitive / non-symmetric relations like this one
        rr_sort(&results, &|l, r| l.0.compare(&r.0))
            // keep the top-k
            [..k]
            // choose one at random
            .choose(&mut rand::thread_rng())
            .map(|(_, action)| action)
            .cloned()
    }
}
