pub mod core;
pub mod games;

use core::evaluator::Evaluator;
use core::minimax::minimax;
use core::position::Position;
use games::connect4::Connect4;
use games::mnk::TicTacToe;

struct NaiveEvaluator;

impl Evaluator<usize, TicTacToe, i32> for NaiveEvaluator {
    fn eval(&self, state: &TicTacToe) -> i32 {
        return 0;
    }
}

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
