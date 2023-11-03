pub mod algos;
pub mod core;
pub mod games;

#[allow(unused_variables)]
#[allow(unused_mut)]
#[allow(unused_imports)]
fn main() {
    use crate::algos::mcts::mcts::{MCTSSpec, MCTS};
    use crate::core::outcome::Outcome;
    use algos::minimax::minimax;
    use core::position::Position;
    use core::result::SearchResult;
    use games::mnk::TicTacToe;
    use thesis::algos::mcts::strategy::DefaultStrategy;
    use thesis::evaluators::null::NullEvaluator;

    let c4 = TicTacToe::initial();
    // TicTacToe::from_string("XX O  X O", 'O');
    println!("{:}", c4);

    /*
    struct YourType;
    impl MCTSSpec for YourType {
        type Position = TicTacToe;
        type Strategy = DefaultStrategy;
    }
    type R = MCTS<YourType>;
    let mut mcts = R::new(&c4, &DefaultStrategy);
    mcts.run_iteration();
    */

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
