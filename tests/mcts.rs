use thesis::algos::mcts::mcts::{MCTSSpec, MCTS};
use thesis::algos::mcts::strategy::DefaultStrategy;
use thesis::algos::minimax::minimax;
use thesis::core::outcome::Outcome;
use thesis::core::position::Position;
use thesis::core::result::SearchResult;
use thesis::evaluators::null::NullEvaluator;
use thesis::games::mnk::TicTacToe;

struct YourType;

impl MCTSSpec for YourType {
    type Position = TicTacToe;
    type Strategy = DefaultStrategy;
}

type R = MCTS<YourType>;

#[test]
fn asd() {
    let board = "XX..O..O.";
    let draw_for_o = TicTacToe::from_str(board, 'O');
    let mut mcts = R::new(&draw_for_o, &DefaultStrategy);

    for i in 1..1000 {
        mcts.run_iteration();
    }

    println!("asdasd");
}
