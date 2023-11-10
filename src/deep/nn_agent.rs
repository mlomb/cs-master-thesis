use super::nn::NNEvaluator;
use crate::core::{agent::Agent, position, result::SearchResult};
use std::{collections::VecDeque, rc::Rc};

pub struct NNAgent {
    evaluator: Rc<NNEvaluator>,
    random_wiggle: VecDeque<usize>,
}

impl NNAgent {
    pub fn new(evaluator: Rc<NNEvaluator>) -> Self {
        NNAgent {
            evaluator,
            random_wiggle: VecDeque::from([8, 8, 8, 4, 3, 2]),
        }
    }
}

impl<Position> Agent<Position> for NNAgent
where
    Position: position::Position,
{
    fn next_action(&mut self, position: &Position) -> Option<Position::Action> {
        let actions = position.valid_actions();

        // we will sample an action from the top-k actions
        // where k is the first element of the random_wiggle
        let k = std::cmp::min(self.random_wiggle.pop_front().unwrap_or(1), actions.len());

        let mut results = Vec::new();

        for action in actions {
            // let (value, _) = alphabeta(&position.apply_action(&action), 3, spec, evaluator);
            let value = SearchResult::<f32>::NonTerminal(0.0);

            results.push((value, action));
        }

        // use an special "sorting" algorithm for non-transitive / non-symmetric relations like this one
        //rr_sort(&results).first().map(|(_, action)| action)

        Some(results.first().unwrap().1.clone())
    }
}
