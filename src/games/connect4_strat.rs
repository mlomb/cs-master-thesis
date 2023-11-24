use super::connect4::{Connect4, COLS, ROWS};
use crate::core::{agent::Agent, evaluator::PositionEvaluator};

pub struct Connect4BasicEvaluator;

fn is_taken(board: [u64; 2], player: usize, row: i32, col: i32) -> bool {
    if row >= 0 && col >= 0 && row < ROWS as i32 && col < COLS as i32 {
        let bit = 1 << (row as usize * COLS + col as usize);

        (board[player] & bit) != 0
    } else {
        false
    }
}

fn count_len(board: [u64; 2], player: usize, len: usize) -> u32 {
    let mut count = 0;
    for row in 0..ROWS as i32 {
        for col in 0..COLS as i32 {
            let mut count_horz = 0 as i32;
            let mut count_vert = 0 as i32;
            let mut count_diag = 0 as i32;
            let mut count_neg_diag = 0 as i32;

            for i in 0..len as i32 {
                count_horz += if is_taken(board, player, row, col + i) {
                    1
                } else {
                    -99
                };
                count_vert += if is_taken(board, player, row + i, col) {
                    1
                } else {
                    -99
                };
                count_diag += if is_taken(board, player, row + i, col + i) {
                    1
                } else {
                    -99
                };
                count_neg_diag += if is_taken(board, player, row + i, col - i) {
                    1
                } else {
                    -99
                };
            }

            count += (count_horz >= len as i32) as u32;
            count += (count_vert >= len as i32) as u32;
            count += (count_diag >= len as i32) as u32;
            count += (count_neg_diag >= len as i32) as u32;
        }
    }
    count
}

fn count(board: [u64; 2], player: usize) -> u32 {
    let mut count = 0;

    count += count_len(board, player, 4) * 1000000;
    count += count_len(board, player, 3) * 1000;
    count += count_len(board, player, 2);

    count
}

impl PositionEvaluator<Connect4, i32> for Connect4BasicEvaluator {
    fn eval(&self, state: &Connect4) -> i32 {
        count(state.0.board, state.0.who_plays as usize) as i32
            - count(state.0.board, (1 - state.0.who_plays) as usize) as i32
    }
}

pub struct Connect4BasicAgent {}

impl Agent<Connect4> for Connect4BasicAgent {
    fn next_action(&mut self, position: &Connect4) -> Option<usize> {
        let (_, action) = crate::search::alphabeta::alphabeta(position, 8, &Connect4BasicEvaluator);

        action
    }
}
