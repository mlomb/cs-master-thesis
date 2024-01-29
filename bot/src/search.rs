use std::ops::Index;

use ndarray::Array2;
use ort::{inputs, Session};
use shakmaty::{Chess, Color, Move, Position};

use crate::encoding::encode_board;

pub fn negamax(chess: Chess, depth: i32, alpha: f32, beta: f32, model: &Session) -> (f32, Move) {
    let moves = chess.legal_moves();
    let mut x = Array2::<i64>::zeros((moves.len(), 12));

    for i in 0..moves.len() {
        let mut chess_moved = chess.clone();
        chess_moved.play_unchecked(&moves[i]);

        if chess_moved.is_checkmate() {
            // detected checkmate
            return (1e6, moves[i].clone());
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

    for index in indexes {
        eprintln!(" - {}: {}", moves[index], scores[index]);

        return (scores[index], moves[index].clone());
    }

    return (0.5, moves[0].clone());

    /*
    for i in 1..legal_moves.len() {
        let mut board_copy = chess.clone();
        board_copy.play_unchecked(&legal_moves[i]);
        boards.push(board_copy);
        moves.push(mv);
        Xs.push(encode_board(&board_copy));
    }

    //let scores = model.run(inputs![x].unwrap()).unwrap()[0]
    //    .extract::<Array2<f32>>()
    //    .unwrap();

    let checkmate_score = 1e6;

    let mut child_nodes: Vec<(f32, Move, Board)> = scores
        .iter()
        .zip(moves.iter())
        .zip(boards.iter())
        .map(|((score, mv), board)| {
            let mut score = -score;

            if board.is_checkmate() {
                score = checkmate_score;
            }

            (score, *mv, board.clone())
        })
        .collect();

    child_nodes.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    let mut best_value = f32::NEG_INFINITY;
    let mut best_move = None;

    for (score, mv, next_board) in child_nodes {
        let value = if depth == 1 || score == checkmate_score {
            score
        } else {
            let (neg_value, _) = negamax(next_board, depth - 1, -beta, -alpha, model);
            -neg_value
        };

        if value > best_value {
            best_value = value;
            best_move = Some(mv);
        }

        if value > alpha {
            alpha = value;
        }

        if alpha > beta {
            break;
        }
    }

    (best_value, best_move.unwrap())
     */
}
