use thesis::algos::negamax::negamax;
use thesis::core::outcome::Outcome::*;
use thesis::core::position::Position;
use thesis::core::result::SearchResult;
use thesis::evaluators::null::NullEvaluator;
use thesis::games::mnk::TicTacToe;

// We need a concrete type for A
type R = SearchResult<i32>;

#[test]
fn draw_on_initial_tictactoe() {
    assert_eq!(
        negamax(&TicTacToe::initial(), 9, &NullEvaluator).0,
        R::Terminal(Draw)
    );
}

#[test]
fn custom_board_tictactoe() {
    let board = "XX..O..O.";
    let win_for_x = TicTacToe::from_str(board, 'X');
    let draw_for_o = TicTacToe::from_str(board, 'O');

    assert_eq!(negamax(&win_for_x, 9, &NullEvaluator).0, R::Terminal(Win));
    assert_eq!(negamax(&draw_for_o, 9, &NullEvaluator).0, R::Terminal(Draw));
}
