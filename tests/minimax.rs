use thesis::algos::minimax::minimax;
use thesis::core::evaluator::{PositionEvaluator, ValueComparator};
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
        minimax(&TicTacToe::initial(), 9, &NullEvaluator, true),
        R::True(Draw)
    );
}

#[test]
fn custom_board_tictactoe() {
    let board = "XX..O..O.";
    let win_for_x = TicTacToe::from_str(board, 'X');
    let draw_for_o = TicTacToe::from_str(board, 'O');

    assert_eq!(minimax(&win_for_x, 9, &NullEvaluator, true), R::True(Win));
    assert_eq!(minimax(&draw_for_o, 9, &NullEvaluator, true), R::True(Draw));
}

#[test]
fn asd() {
    struct BoardEvaluator;

    impl ValueComparator<TicTacToe> for BoardEvaluator {
        fn is_better(&self, _: &TicTacToe, _: &TicTacToe) -> bool {
            panic!("NullEvaluator should never compare values")
        }
    }

    impl PositionEvaluator<TicTacToe, TicTacToe> for BoardEvaluator {
        fn eval(&self, state: &TicTacToe) -> TicTacToe {
            state.clone()
        }
    }

    let board = "XX..O..O.";
    let win_for_x = TicTacToe::from_str(board, 'X');
    let draw_for_o = TicTacToe::from_str(board, 'O');

    assert_eq!(minimax(&win_for_x, 9, &NullEvaluator, true), R::True(Win));
    assert_eq!(minimax(&draw_for_o, 9, &NullEvaluator, true), R::True(Draw));
}
