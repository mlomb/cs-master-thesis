// Minimax

use crate::core::position::*;

pub fn minimax<Action, P: Position<Action>>(position: &P, max_depth: usize) -> i32 {
    if max_depth == 0 {
        return 0;
    }

    // println!("{:}", position);
    // dbg!(position.status());

    match position.status() {
        Status::LOSS => return -1,
        Status::DRAW => return 0,
        Status::WIN => return 1,
        Status::PLAYING => {}
    }

    let mut best_score = i32::MIN;

    for action in position.valid_actions() {
        let score = -minimax(&position.apply_action(action), max_depth - 1);
        best_score = best_score.max(score);
    }

    best_score
}
