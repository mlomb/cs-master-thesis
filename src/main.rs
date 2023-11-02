pub mod algos;
pub mod core;
pub mod games;

use algos::minimax::minimax;
use core::evaluator::Evaluator;
use core::position::Position;
use games::mnk::TicTacToe;

struct NaiveEvaluator;

impl Evaluator<usize, TicTacToe, i32> for NaiveEvaluator {
    fn eval(&self, state: &TicTacToe) -> i32 {
        return state.board[0] as i32 - state.board[1] as i32;
    }
}

#[allow(unused_variables)]
#[allow(unused_mut)]
fn main() {
    println!("Hello, world!");

    let mut c4 = TicTacToe::initial();
    // TicTacToe::from_string("XX O  X O", 'O');
    println!("{:?}", c4);

    let res = minimax(&c4, 150, &NaiveEvaluator);
    dbg!(res);

    /*
    println!("{:?}", c4.status());
    c4 = c4.apply_action(3);
    c4 = c4.apply_action(3);
    c4 = c4.apply_action(4);
    c4 = c4.apply_action(4);
    c4 = c4.apply_action(5);
    c4 = c4.apply_action(5);
    c4 = c4.apply_action(6);
    //c4 = c4.apply_action(6);
    println!("{:?}", c4.status());
    */
}
