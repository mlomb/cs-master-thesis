use std::ops::Index;

use ndarray::Array2;
use ort::{inputs, Session};
use shakmaty::{Chess, Color, Move, Position};

use crate::encoding::encode_board;

pub fn negamax(
    chess: Chess,
    depth: i32,
    mut alpha: f32,
    beta: f32,
    model: &Session,
) -> (f32, Option<Move>) {
    if chess.is_stalemate() {
        return (0.0, None);
    }

    let moves = chess.legal_moves();
    let mut x = Array2::<i64>::zeros((moves.len(), 12));

    assert!(moves.len() > 0, "no legal moves for position: {:?}", chess);

    for i in 0..moves.len() {
        let mut chess_moved = chess.clone();
        chess_moved.play_unchecked(&moves[i]);

        if chess_moved.is_checkmate() {
            // detected checkmate
            return (1e6, Some(moves[i].clone()));
        }

        let mut board_moved = chess_moved.board().clone();
        if chess_moved.turn() == Color::Black {
            board_moved.flip_vertical();
            board_moved.swap_colors();
        }

        let mut row = x.row_mut(i);
        let encoded = encode_board(&board_moved);
        for j in 0..12 {
            row[j] = encoded[j];
        }
    }

    let outputs = model.run(inputs![x].unwrap()).unwrap();
    let scores = outputs.index(0).extract_raw_tensor::<f32>().unwrap().1;

    let mut indexes: Vec<usize> = (0..moves.len()).collect();
    // sort from LOWEST to HIGHEST
    indexes.sort_by(|a, b| scores[*a].partial_cmp(&scores[*b]).unwrap());

    let mut best_value = f32::NEG_INFINITY;
    let mut best_move = None;

    for index in indexes {
        let value: f32;

        if depth == 1 {
            // since the move has been made, and the position
            // has been evaluated for the other player,
            // we have to negate the score
            // this is why we sort from LOWEST to HIGHEST
            value = -scores[index];
        } else {
            let mut chess_moved = chess.clone();
            chess_moved.play_unchecked(&moves[index]);

            let (neg_value, _) = negamax(chess_moved, depth - 1, -beta, -alpha, model);
            value = -neg_value;
        }

        // eprintln!(" - {}: {}", moves[index], scores[index]);

        if value > best_value {
            best_value = value;
            best_move = Some(moves[index].clone());
        }

        if value > alpha {
            alpha = value;
        }

        if alpha > beta {
            break;
        }
    }

    return (best_value, best_move);
}
