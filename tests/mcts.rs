use thesis::algos::mcts::mcts::MCTS;
use thesis::algos::mcts::strategy::DefaultStrategy;
use thesis::algos::minimax::minimax;
use thesis::core::outcome::Outcome;
use thesis::core::position::Position;
use thesis::core::result::SearchResult;
use thesis::evaluators::null::NullEvaluator;
use thesis::games::mnk::TicTacToe;

type M = MCTS<usize, TicTacToe, DefaultStrategy>;

#[test]
fn asd() {
    let board = "XX..O..O.";
    let draw_for_o = TicTacToe::from_str(board, 'O');
    let mut mcts = M::new(&draw_for_o, &DefaultStrategy);

    for i in 1..1000 {
        mcts.run_iteration();
    }

    println!("asdasd");
}
