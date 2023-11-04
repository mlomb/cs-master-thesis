use crate::core::evaluator;
use crate::core::position;
use crate::core::result::SearchResult;

pub fn minimax<
    Position: position::Position,
    Value,
    Evaluator: evaluator::PositionEvaluator<Position, Value> + evaluator::ValueComparator<Value>,
>(
    position: &Position,
    max_depth: usize,
    evaluator: &Evaluator,
    maximizing: bool,
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
        let branch_result = minimax(
            &position.apply_action(&action),
            max_depth - 1,
            evaluator,
            !maximizing,
        );

        best = match best {
            None => Some(branch_result),
            Some(current_best)
                if maximizing && branch_result.is_better_than(&current_best, evaluator) =>
            {
                Some(branch_result)
            }
            _ => best,
        };
    }

    best.expect("should have at least one valid action")
}
