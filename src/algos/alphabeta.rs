use crate::core::evaluator;
use crate::core::position;
use crate::core::result::SearchResult;
use crate::core::value;
use std::cmp::Ordering::*;
use std::fmt::Debug;

fn alphabeta_impl<Position, Value, ValuePolicy, Evaluator>(
    position: &Position,
    max_depth: usize,
    alpha: &Option<SearchResult<Value>>,
    beta: &Option<SearchResult<Value>>,
    value_policy: &mut ValuePolicy,
    evaluator: &Evaluator,
) -> (SearchResult<Value>, Option<Position::Action>)
where
    Position: position::Position,
    Value: Clone + Debug,
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
    let mut alpha = alpha.clone();

    for action in position.valid_actions() {
        let (opp_branch_result, _) = alphabeta_impl(
            &position.apply_action(&action),
            max_depth - 1,
            // intentionally swapped
            &beta.as_ref().and_then(|a| Some(a.opposite(value_policy))),
            &alpha.as_ref().and_then(|a| Some(a.opposite(value_policy))),
            value_policy,
            evaluator,
        );
        let branch_result = opp_branch_result.opposite(value_policy);

        if match alpha {
            None => true,
            Some(ref alpha_res) => branch_result.compare(&alpha_res, value_policy) == Greater,
        } {
            alpha = Some(branch_result.clone());
        }

        if match best {
            None => true,
            Some((ref best_res, _)) => branch_result.compare(&best_res, value_policy) == Greater,
        } {
            best = Some((branch_result, Some(action)));
        }

        // cutoff
        if let Some(ref beta_res) = beta {
            if alpha.as_ref().unwrap().compare(&beta_res, value_policy) >= Equal {
                // early out
                return (alpha.unwrap(), None);
            }
        }
    }

    best.expect("should have at least one valid action")
}

pub fn alphabeta<Position, Value, ValuePolicy, Evaluator>(
    position: &Position,
    max_depth: usize,
    value_policy: &mut ValuePolicy,
    evaluator: &Evaluator,
) -> (SearchResult<Value>, Option<Position::Action>)
where
    Position: position::Position,
    Value: Clone + Debug,
    ValuePolicy: value::ValuePolicy<Value>,
    Evaluator: evaluator::PositionEvaluator<Position, Value>,
{
    alphabeta_impl(position, max_depth, &None, &None, value_policy, evaluator)
}
