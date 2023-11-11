use rand::seq::SliceRandom;

use crate::{
    algos::alphabeta::alphabeta,
    core::{agent::Agent, position, result::SearchResult},
    nn::nn_encoding::TensorEncodeable,
};
use std::{collections::VecDeque, rc::Rc};

use super::evaluator::DeepCmpEvaluator;

pub struct DeepCmpAgent<Position> {
    evaluator: Rc<DeepCmpEvaluator<Position>>,
    random_wiggle: VecDeque<usize>,
}

impl<Position> DeepCmpAgent<Position> {
    pub fn new(evaluator: Rc<DeepCmpEvaluator<Position>>) -> Self {
        DeepCmpAgent {
            evaluator,
            random_wiggle: VecDeque::from([8, 8, 8, 4, 3, 2]),
        }
    }
}

impl<Position> Agent<Position> for DeepCmpAgent<Position>
where
    Position: position::Position + TensorEncodeable,
{
    fn next_action(&mut self, position: &Position) -> Option<Position::Action> {
        let actions = position.valid_actions();

        // we will sample an action from the top-k actions
        // where k is the first element of the random_wiggle
        let k = std::cmp::min(self.random_wiggle.pop_front().unwrap_or(1), actions.len());

        let mut results = Vec::new();

        for action in actions {
            // let (value, _) = alphabeta::<_, _, NNEvaluator>(
            //     &position.apply_action(&action),
            //     3,
            //     self.evaluator.as_ref(),
            // );
            let value = SearchResult::NonTerminal(0.0);

            results.push((value, action));
        }

        // use an special "sorting" algorithm for non-transitive / non-symmetric relations like this one
        //rr_sort(&results).first().map(|(_, action)| action)

        results[..k]
            .choose(&mut rand::thread_rng())
            .map(|(_, action)| action)
            .cloned()
    }
}
