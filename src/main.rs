pub mod algos;
pub mod core;
pub mod games;

use algos::minimax::minimax;
use core::position::Position;
use core::result::SearchResult;
use games::mnk::TicTacToe;
use thesis::evaluators::null::NullEvaluator;

#[allow(unused_variables)]
#[allow(unused_mut)]
fn main() {
    println!("Hello, world!");

    let mut c4 = TicTacToe::initial();
    // TicTacToe::from_string("XX O  X O", 'O');
    println!("{:?}", c4);

    let res: SearchResult<i32> = minimax(&c4, 150, &NullEvaluator);
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
