use crate::core::evaluator;
use crate::core::position;
use crate::core::result::SearchResult;
use crate::core::value;
use std::cmp::Ordering::*;

pub fn negamax<Position, Value, ValuePolicy, Evaluator>(
    position: &Position,
    max_depth: usize,
    value_policy: &mut ValuePolicy,
    evaluator: &Evaluator,
) -> (SearchResult<Value>, Option<Position::Action>)
where
    Position: position::Position,
    ValuePolicy: value::ValuePolicy<Value>,
    Evaluator: evaluator::PositionEvaluator<Position, Value>,
{
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
        let (opp_branch_result, _) = negamax(
            &position.apply_action(&action),
            max_depth - 1,
            value_policy,
            evaluator,
        );
        let branch_result = opp_branch_result.opposite(value_policy);

        if match best {
            None => true,
            Some((ref res, _)) => branch_result.compare(&res, value_policy) == Greater,
        } {
            best = Some((branch_result, Some(action)))
        }
    }

    best.expect("should have at least one valid action")
}