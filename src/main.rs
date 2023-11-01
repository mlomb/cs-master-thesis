pub mod core;
pub mod games;

use core::eval::Eval;

use crate::core::minimax::minimax;
use crate::core::position::Position;
use crate::games::connect4::Connect4;
use crate::games::mnk::TicTacToe;

struct NaiveEvaluator;

impl<A, P: Position<A>> Eval<A, P, i32> for NaiveEvaluator {
    fn eval(state: P) -> i32 {
        return 0;
    }

    fn terminal_value(&self) -> i32 {
        todo!()
    }
}

fn main() {
    println!("Hello, world!");

    let mut c4 = Connect4::initial();
    // TicTacToe::from_string("XX O  X O", 'O');
    println!("{:?}", c4);

    let res = minimax(&c4, 7);
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
