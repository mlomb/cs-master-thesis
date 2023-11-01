// Minimax

use super::evaluator::Evaluator;
use crate::core::position::*;

pub fn minimax<A, P: Position<A>, V, E: Evaluator<A, P, V>>(
    position: &P,
    max_depth: usize,
    evaluator: &E,
) -> V {
    if max_depth == 0 {
        return evaluator.eval(&position);
    }

    // println!("{:}", position);
    // dbg!(position.status());

    match position.status() {
        // Status::LOSS => return -1,
        // Status::DRAW => return 0,
        // Status::WIN => return 1,
        Status::PLAYING => {}
        _ => return evaluator.eval(&position),
    }

    let mut best_score = i32::MIN;

    for action in position.valid_actions() {
        let score = -minimax(&position.apply_action(action), max_depth - 1, evaluator);
        best_score = best_score.max(score);
    }

    best_score
}
