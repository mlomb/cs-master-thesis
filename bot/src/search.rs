use crate::{encoding::encode_board, pv::PVTable};
use ndarray::Array2;
use ort::{inputs, Session};
use shakmaty::{CastlingMode, Chess, Color, Move, MoveList, Position};
use std::{
    ops::Index,
    time::{Duration, Instant},
};

const INFINITE: f32 = 50000.0;

pub struct Search {
    /// ML model to evaluate positions
    model: Session,

    /// Number of nodes searched
    nodes: usize,
    /// Number of evals computed
    evals: usize,

    /// Current ply
    ply: usize,
    /// Principal variation
    pv: PVTable,

    /// Start search time
    start_time: Instant,
    /// Time limit, do not exceed this time
    time_limit: Option<Duration>,
    /// Whether the search was aborted
    aborted: bool,
}

impl Search {
    pub fn new(model: Session) -> Self {
        Search {
            model,
            nodes: 0,
            evals: 0,
            start_time: Instant::now(),
            ply: 0,
            pv: PVTable::new(),
            time_limit: None,
            aborted: false,
        }
    }

    pub fn go(
        &mut self,
        position: &Chess,
        max_depth: Option<i32>,
        time_limit: Option<Duration>,
    ) -> Option<Move> {
        self.nodes = 0;
        self.evals = 0;
        self.ply = 0;

        // time control
        self.start_time = Instant::now();
        self.time_limit = time_limit;
        self.aborted = false;

        let mut best_score = None;
        let mut best_line = None;

        for depth in 1..=max_depth.unwrap_or(2) {
            // try aspiration window close to the previous iteration's score
            // we assume that the score of this iteration will not be changing much
            const DELTA: f32 = 0.12;
            let mut alpha = (best_score.unwrap_or(-INFINITE) - DELTA).max(-INFINITE);
            let mut beta = (best_score.unwrap_or(INFINITE) + DELTA).min(INFINITE);
            let mut score;
            alpha = -INFINITE;
            beta = INFINITE;

            loop {
                score = self.negamax(position.clone(), alpha, beta, depth);

                // if failing high/low, re-search the position
                if score < alpha || score > beta {
                    // failing high/low, re-search the position with the full window
                    // score is outside the window
                    alpha = -INFINITE;
                    beta = INFINITE;
                    eprintln!("fail high/low, re-searching with full window");
                } else {
                    // score is within the window
                    break;
                }
            }

            if self.aborted {
                break;
            }

            best_line = Some(self.pv.get_mainline());
            best_score = Some(score);

            eprint!(
                "info depth {} nodes {} evals {} time {} score cp {} pv ",
                depth,
                self.nodes,
                self.evals,
                self.start_time.elapsed().as_millis(),
                score
            );
            for mv in self.pv.get_mainline() {
                eprint!("{} ", mv.to_uci(CastlingMode::Standard).to_string());
            }
            eprint!("\n");
        }

        best_line.iter().next()
    }

    fn quiescence(&mut self, position: Chess, mut alpha: f32, beta: f32) -> f32 {
        // increment the number of nodes searched
        self.nodes += 1;

        // evaluate the position
        let score = self.evaluate(&position);

        // fail-hard beta cutoff
        if score >= beta {
            // node (move) fails high
            return beta;
        }

        // found a better move
        if score > alpha {
            // PV node (move)
            alpha = score;
        }

        // only look at captures
        for move_ in position.capture_moves() {
            let mut chess_moved = position.clone();
            chess_moved.play_unchecked(&move_);

            self.ply += 1;
            let score = -self.quiescence(chess_moved, -beta, -alpha);
            self.ply -= 1;

            // fail-hard beta cutoff
            if score >= beta {
                // node (move) fails high
                return beta;
            }

            // found a better move
            if score > alpha {
                // PV node (move)
                alpha = score;
            }
        }

        // node (move) fails low
        return alpha;
    }

    fn negamax(&mut self, position: Chess, mut alpha: f32, beta: f32, depth: i32) -> f32 {
        if self.nodes % 1000 == 0 {
            self.checkup();
        }

        if self.aborted {
            // abort search (time limit?)
            return 0.0;
        }

        // init PV
        let mut found_pv = false;
        self.pv.reset(self.ply);

        // TODO: optimize
        if position.is_stalemate() {
            return 0.0;
        }

        if position.is_checkmate() {
            return -10000.0;
        }

        if depth == 0 {
            // escape from recursion
            // run quiescence search
            //return self.quiescence(position, alpha, beta);
            self.nodes += 1;
            return self.evaluate(&position);
        }

        // increment the number of nodes searched
        self.nodes += 1;

        // generate legal moves
        let mut moves = position.legal_moves();
        assert!(moves.len() > 0, "at least one legal move");

        // sort moves
        self.sort_moves(&mut moves);

        for mv in moves {
            let mut chess_moved = position.clone();
            chess_moved.play_unchecked(&mv);

            self.ply += 1;

            let mut score;

            // variable to store current move's score
            if found_pv {
                /* Once you've found a move with a score that is between alpha and beta,
                the rest of the moves are searched with the goal of proving that they are all bad.
                It's possible to do this a bit faster than a search that worries that one
                of the remaining moves might be good. */
                score = -self.negamax(chess_moved.clone(), -alpha - 0.0001, -alpha, depth - 1);

                /* If the algorithm finds out that it was wrong, and that one of the
                subsequent moves was better than the first PV move, it has to search again,
                in the normal alpha-beta manner.  This happens sometimes, and it's a waste of time,
                but generally not often enough to counteract the savings gained from doing the
                "bad move proof" search referred to earlier. */
                if (score > alpha) && (score < beta) {
                    // Check for failure.
                    /* re-search the move that has failed to be proved to be bad
                    with normal alpha beta score bounds*/
                    score = -self.negamax(chess_moved, -beta, -alpha, depth - 1);
                }
            } else {
                // for all other types of nodes
                // do normal alpha beta search
                score = -self.negamax(chess_moved, -beta, -alpha, depth - 1);
            };

            self.ply -= 1;

            eprintln!("alpha={} beta={} score={}", alpha, beta, score);

            // fail-hard beta cutoff
            if score >= beta {
                // TODO: killer moves

                // fails high
                return beta;
            }

            // found a better move!
            if score > alpha {
                // PV node
                alpha = score;

                // enable found PV
                found_pv = true;

                // write the PV table
                self.pv.write(self.ply, mv);
            }
        }

        // TODO: checkmate?

        // node fails low
        alpha
    }

    fn sort_moves(&self, moves: &mut MoveList) {
        // sort using the NN (too slow)
        // moves.sort_by_cached_key(|mv| {
        //     let mut chess_moved = position.clone();
        //     chess_moved.play_unchecked(&mv);
        //     -(self.evaluate(&chess_moved) * 100000.0) as i32
        // });

        let move_pv = self.pv.get_best_move(self.ply);

        // sort using an heuristic
        moves.sort_by_cached_key(|mv| {
            let mut score = 0;

            // put the PV move first
            if mv == move_pv {
                score = 20_000;
            }

            if mv.is_capture() {
                // MVV LVA [attacker][victim]
                const MVV_LVA: [[i32; 12]; 12] = [
                    [105, 205, 305, 405, 505, 605, 105, 205, 305, 405, 505, 605],
                    [104, 204, 304, 404, 504, 604, 104, 204, 304, 404, 504, 604],
                    [103, 203, 303, 403, 503, 603, 103, 203, 303, 403, 503, 603],
                    [102, 202, 302, 402, 502, 602, 102, 202, 302, 402, 502, 602],
                    [101, 201, 301, 401, 501, 601, 101, 201, 301, 401, 501, 601],
                    [100, 200, 300, 400, 500, 600, 100, 200, 300, 400, 500, 600],
                    // --
                    [105, 205, 305, 405, 505, 605, 105, 205, 305, 405, 505, 605],
                    [104, 204, 304, 404, 504, 604, 104, 204, 304, 404, 504, 604],
                    [103, 203, 303, 403, 503, 603, 103, 203, 303, 403, 503, 603],
                    [102, 202, 302, 402, 502, 602, 102, 202, 302, 402, 502, 602],
                    [101, 201, 301, 401, 501, 601, 101, 201, 301, 401, 501, 601],
                    [100, 200, 300, 400, 500, 600, 100, 200, 300, 400, 500, 600],
                ];

                let attacker = mv.role();
                let victim = mv.capture().unwrap();

                score = MVV_LVA[attacker as usize][victim as usize] + 10_000;
            }

            // sorts are from low to high, so flip
            -score
        });
    }

    fn evaluate(&mut self, position: &Chess) -> f32 {
        // increase number of evals computed
        self.evals += 1;

        return Search::basic_eval(position);

        //let hash: Zobrist32 = position.zobrist_hash(shakmaty::EnPassantMode::Always);
        //return hash.0 as f32 / u32::MAX as f32;

        let mut board = position.board().clone();
        if position.turn() == Color::Black {
            board.flip_vertical();
            board.swap_colors();
        }

        let encoded = encode_board(&board);
        let mut x = Array2::<i64>::zeros((1, 12));
        let mut row = x.row_mut(0);

        for j in 0..12 {
            row[j] = encoded[j];
        }

        unsafe {
            let outputs = self
                .model
                .run(inputs![x].unwrap_unchecked())
                .unwrap_unchecked();
            let scores = outputs
                .index(0)
                .extract_raw_tensor::<f32>()
                .unwrap_unchecked()
                .1;
            let value = scores[0];

            return value;
        }
    }

    fn basic_eval(position: &Chess) -> f32 {
        use shakmaty::Role::*;

        // pawn positional score
        const PAWN_SCORE: [i32; 64] = [
            90, 90, 90, 90, 90, 90, 90, 90, 30, 30, 30, 40, 40, 30, 30, 30, 20, 20, 20, 30, 30, 30,
            20, 20, 10, 10, 10, 20, 20, 10, 10, 10, 5, 5, 10, 20, 20, 5, 5, 5, 0, 0, 0, 5, 5, 0, 0,
            0, 0, 0, 0, -10, -10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ];
        const KNIGHT_SCORE: [i32; 64] = [
            -5, 0, 0, 0, 0, 0, 0, -5, -5, 0, 0, 10, 10, 0, 0, -5, -5, 5, 20, 20, 20, 20, 5, -5, -5,
            10, 20, 30, 30, 20, 10, -5, -5, 10, 20, 30, 30, 20, 10, -5, -5, 5, 20, 10, 10, 20, 5,
            -5, -5, 0, 0, 0, 0, 0, 0, -5, -5, -10, 0, 0, 0, 0, -10, -5,
        ];
        const BISHOP_SCORE: [i32; 64] = [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 10, 0, 0, 0, 0, 0, 10, 20,
            20, 10, 0, 0, 0, 0, 10, 20, 20, 10, 0, 0, 0, 10, 0, 0, 0, 0, 10, 0, 0, 30, 0, 0, 0, 0,
            30, 0, 0, 0, -10, 0, 0, -10, 0, 0,
        ];
        const ROOK_SCORE: [i32; 64] = [
            50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 0, 0, 10, 20, 20, 10,
            0, 0, 0, 0, 10, 20, 20, 10, 0, 0, 0, 0, 10, 20, 20, 10, 0, 0, 0, 0, 10, 20, 20, 10, 0,
            0, 0, 0, 10, 20, 20, 10, 0, 0, 0, 0, 0, 20, 20, 0, 0, 0,
        ];
        const KING_SCORE: [i32; 64] = [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 0, 0, 0, 5, 5, 10, 10, 5, 5, 0, 0, 5, 10, 20,
            20, 10, 5, 0, 0, 5, 10, 20, 20, 10, 5, 0, 0, 0, 5, 10, 10, 5, 0, 0, 0, 5, 5, -5, -5, 0,
            5, 0, 0, 0, 5, 0, -15, 0, 10, 0,
        ];

        let mut score = 0;

        for (mut square, piece) in position.board().clone().into_iter() {
            let material_score = match piece.role {
                Pawn => 100,
                Knight => 300,
                Bishop => 350,
                Rook => 500,
                Queen => 1000,
                King => 10000,
            };

            if piece.color == Color::Black {
                square = square.flip_vertical();
            }

            let positional_score = match piece.role {
                Pawn => PAWN_SCORE[square as usize],
                Knight => KNIGHT_SCORE[square as usize],
                Bishop => BISHOP_SCORE[square as usize],
                Rook => ROOK_SCORE[square as usize],
                Queen => 0,
                King => KING_SCORE[square as usize],
            };

            let sign = if piece.color == position.turn() {
                1
            } else {
                -1
            };

            score += sign * (material_score + positional_score);
        }

        score as f32
    }

    fn checkup(&mut self) {
        if let Some(time_limit) = self.time_limit {
            if self.start_time.elapsed() >= time_limit {
                self.aborted = true;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_mates() {
        const MATE_FENS: [&str; 1] = [
            //
            "4k3/4N2R/8/1P1K2P1/8/8/PP5P/8 w - - 5 48",
        ];
    }
}
