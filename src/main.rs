use crate::algos::mcts::MCTS;

pub mod algos;
pub mod core;
pub mod games;

#[allow(unused_variables)]
#[allow(unused_mut)]
#[allow(unused_imports)]
fn main() {
    use crate::core::outcome::Outcome;
    use algos::minimax::minimax;
    use core::position::Position;
    use core::result::SearchResult;
    use games::mnk::TicTacToe;
    use thesis::evaluators::null::NullEvaluator;

    println!("Hello, world!");

    let c4 = TicTacToe::initial();
    // TicTacToe::from_string("XX O  X O", 'O');
    println!("{:?}", c4);

    let mut mcts = MCTS::new(&c4);
    mcts.run_iteration();

    //let res: SearchResult<i32> = minimax(&c4, 150, &NullEvaluator);
    //dbg!(res);

    //type R = SearchResult<i32>;
    //assert_eq!(minimax(&c4, 9, &NullEvaluator), R::True(Outcome::Win));

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
