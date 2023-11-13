use std::collections::HashSet;
use std::hash::Hash;
use thesis::core::position::Position;
use thesis::core::result::SearchResult;
use thesis::evaluators::null::NullEvaluator;
use thesis::games::connect4::Connect4;
use thesis::games::connect4_strat::Connect4BasicEvaluator;
use thesis::games::mnk::TicTacToe;
use thesis::search::alphabeta::alphabeta;
use thesis::search::negamax::negamax;

fn generate_boards<P: Position + Sized + Eq + Hash>(
    from: &P,
    depth: usize,
    vec: &mut Vec<P>,
    hs: &mut HashSet<P>,
) {
    if depth == 0 || from.status().is_some() {
        return;
    }

    for action in from.valid_actions() {
        let next = from.apply_action(&action);
        if hs.contains(&next) {
            continue;
        }

        hs.insert(next.clone());
        generate_boards(&next, depth - 1, vec, hs);
        vec.push(next);
    }
}

#[test]
fn compare_with_negamax_ttt() {
    let mut boards = vec![TicTacToe::initial()];
    let mut hs = HashSet::<TicTacToe>::new();
    generate_boards(&TicTacToe::initial(), 10, &mut boards, &mut hs);

    // https://math.stackexchange.com/a/486548
    assert_eq!(boards.len(), 5478);

    let mut ne = NullEvaluator;

    type R = (SearchResult<i32>, Option<usize>);

    for board in boards {
        let (nega_res, _): R = negamax(&board, 10, &mut ne);
        let (alphabeta_res, _): R = alphabeta(&board, 10, &mut ne);

        assert_eq!(nega_res, alphabeta_res);
    }
}

#[test]
fn compare_with_negamax_c4() {
    let mut boards = vec![Connect4::initial()];
    let mut hs = HashSet::<Connect4>::new();
    generate_boards(&Connect4::initial(), 2, &mut boards, &mut hs);

    let be = Connect4BasicEvaluator;

    type R = (SearchResult<i32>, Option<usize>);

    for board in boards {
        let (nega_res, _): R = negamax(&board, 4, &be);
        let (alphabeta_res, _): R = alphabeta(&board, 4, &be);

        assert_eq!(nega_res, alphabeta_res);
    }
}
