use crate::core::evaluator;
use crate::core::position;
use crate::core::result::SearchResult;
use crate::core::value;

pub fn minimax<
    Action,
    Position: position::Position<Action>,
    Value: value::Value,
    Evaluator: evaluator::Evaluator<Action, Position, Value>,
>(
    position: &Position,
    max_depth: usize,
    evaluator: &Evaluator,
) -> SearchResult<Value> {
    // If the game is over, return the true outcome
    if let Some(outcome) = position.status() {
        return SearchResult::True(outcome);
    }

    // If we've reached the maximum depth, return the evaluation
    if max_depth == 0 {
        return SearchResult::Eval(evaluator.eval(&position));
    }

    let mut best: Option<SearchResult<Value>> = None;

    for action in position.valid_actions() {
        let branch_result = minimax(&position.apply_action(&action), max_depth - 1, evaluator);

        best = match best {
            None => Some(branch_result),
            Some(current_best)
                if branch_result.compare(&current_best) == std::cmp::Ordering::Greater =>
            {
                Some(branch_result)
            }
            _ => best,
        };
    }

    best.expect("should have at least one valid action")
}
