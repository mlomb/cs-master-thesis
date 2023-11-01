use std::ops::Neg;

use crate::core::evaluator::Evaluator;
use crate::core::position::*;
use crate::core::result::SearchResult;

/*
pub fn minimax<A, P: Position<A>, V: Neg, E: Evaluator<A, P, V>>(
    position: &P,
    max_depth: usize,
    evaluator: &E,
) -> SearchResult<V> {
    if max_depth == 0 {
        return SearchResult::Eval(evaluator.eval(&position));
    }

    // println!("{:}", position);
    // dbg!(position.status());

    match position.status() {
        Status::PLAYING => {
            let mut best_score: SearchResult<V>;

            for action in position.valid_actions() {
                let score = -minimax(&position.apply_action(action), max_depth - 1, evaluator);
                best_score = best_score.max(score);
            }

            best_score
        }
        status => SearchResult::Final(status),
    }
}
 */
pub fn minimax() {}
