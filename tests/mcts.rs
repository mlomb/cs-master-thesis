use thesis::algos::mcts::mcts::{MCTSSpec, MCTS};
use thesis::algos::mcts::strategy::DefaultStrategy;
use thesis::core::position::Position;
use thesis::games::mnk::TicTacToe;

struct YourType;

impl MCTSSpec for YourType {
    type Position = TicTacToe;
    type Strategy = DefaultStrategy;
}

type R = MCTS<YourType>;

#[test]
fn asd() {
    let mut position = TicTacToe::from_str("XX..O..O.", 'O');
    let mut mcts = R::new(&position, &DefaultStrategy);

    while let None = position.status() {
        for _ in 1..1000 {
            mcts.run_iteration();
        }

        let distr = mcts.get_action_distribution();
        let best_action = mcts.get_best_action();

        println!("board: \n{:}", position);
        println!("best action: {:?}", best_action);
        println!("distribution: {:?}", distr);

        mcts.move_root(best_action.unwrap());
        position = position.apply_action(&best_action.unwrap());

        println!("after best action: \n{:}", position);
    }

    println!("Result: {:?}", position.status());
}
