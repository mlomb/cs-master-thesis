use thesis::algos::minimax::minimax;
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
        minimax(&TicTacToe::initial(), 9, &NullEvaluator),
        R::True(Draw)
    );
}

#[test]
fn custom_board_tictactoe() {
    let board = "XX..O..O.";
    let win_for_x = TicTacToe::from_str(board, 'X');
    let draw_for_o = TicTacToe::from_str(board, 'O');

    assert_eq!(minimax(&win_for_x, 9, &NullEvaluator), R::True(Win));
    assert_eq!(minimax(&draw_for_o, 9, &NullEvaluator), R::True(Draw));
}
