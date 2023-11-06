use crate::core::evaluator;
use crate::core::position;
use crate::core::result::SearchResult;
use crate::core::value;

pub fn minimax<
    Position: position::Position,
    Value,
    ValuePolicy: value::ValuePolicy<Value>,
    Evaluator: evaluator::PositionEvaluator<Position, Value>,
>(
    position: &Position,
    max_depth: usize,
    value_policy: &mut ValuePolicy,
    evaluator: &Evaluator,
) -> (SearchResult<Value>, Option<Position::Action>) {
    // If the game is over, return the true outcome
    if let Some(outcome) = position.status() {
        return (SearchResult::Terminal(outcome), None);
    }

    // If we've reached the maximum depth, return the evaluation
    if max_depth == 0 {
        return (SearchResult::NonTerminal(evaluator.eval(&position)), None);
    }

    let mut best: Option<(SearchResult<Value>, Option<Position::Action>)> = None;

    for action in position.valid_actions() {
        let (mut branch_result, _) = minimax(
            &position.apply_action(&action),
            max_depth - 1,
            value_policy,
            evaluator,
        );
        branch_result = branch_result.opposite(value_policy);

        best = match best {
            None => Some((branch_result, Some(action))),
            Some((current_best, _))
                if branch_result.compare(&current_best, value_policy)
                    == std::cmp::Ordering::Greater =>
            {
                Some((branch_result, Some(action)))
            }
            _ => best,
        };
    }

    best.expect("should have at least one valid action")
}
