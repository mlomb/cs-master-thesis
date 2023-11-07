use thesis::algos::negamax::negamax;
use thesis::core::evaluator::PositionEvaluator;
use thesis::core::outcome::Outcome::*;
use thesis::core::position::Position;
use thesis::core::result::SearchResult;
use thesis::core::value::DefaultValuePolicy;
use thesis::evaluators::null::NullEvaluator;
use thesis::games::mnk::TicTacToe;

// We need a concrete type for A
type R = SearchResult<i32>;

#[test]
fn draw_on_initial_tictactoe() {
    assert!(matches!(
        negamax(
            &TicTacToe::initial(),
            9,
            &mut DefaultValuePolicy,
            &NullEvaluator
        ),
        (R::Terminal(Draw), _),
    ));
}

#[test]
fn custom_board_tictactoe() {
    let board = "XX..O..O.";
    let win_for_x = TicTacToe::from_str(board, 'X');
    let draw_for_o = TicTacToe::from_str(board, 'O');

    assert!(matches!(
        negamax(&win_for_x, 9, &mut DefaultValuePolicy, &NullEvaluator),
        (R::Terminal(Win), _),
    ));
    assert!(matches!(
        negamax(&draw_for_o, 9, &mut DefaultValuePolicy, &NullEvaluator),
        (R::Terminal(Draw), _),
    ));
}

#[test]
fn asd() {
    struct BoardEvaluator;

    impl PositionEvaluator<TicTacToe, TicTacToe> for BoardEvaluator {
        fn eval(&self, state: &TicTacToe) -> TicTacToe {
            state.clone()
        }
    }

    let board = "XX..O..O.";
    let win_for_x = TicTacToe::from_str(board, 'X');
    let draw_for_o = TicTacToe::from_str(board, 'O');

    assert!(matches!(
        negamax(&win_for_x, 9, &mut DefaultValuePolicy, &NullEvaluator),
        (R::Terminal(Win), _)
    ));
    assert!(matches!(
        negamax(&draw_for_o, 9, &mut DefaultValuePolicy, &NullEvaluator),
        (R::Terminal(Draw), _),
    ));
}
