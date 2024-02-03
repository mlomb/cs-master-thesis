use crate::{encoding::encode_board, pv::PVTable};
use ndarray::Array2;
use ort::{inputs, Session};
use shakmaty::{CastlingMode, Chess, Color, Move, Position};
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
    /// Whether to follow the principal variation
    // follow_pv: bool,
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
            // follow_pv: false,
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
        let mut best_move = None;

        for depth in 1..=max_depth.unwrap_or(60) {
            // enable following the PV
            // self.follow_pv = true;

            let mut delta = 0.12;
            let mut alpha = (best_score.unwrap_or(-INFINITE) - delta).max(-INFINITE);
            let mut beta = (best_score.unwrap_or(INFINITE) + delta).min(INFINITE);
            let mut score;

            // start with a small aspiration window
            // if the score is outside the window, re-search the position
            // until we dont fail anymore
            loop {
                score = self.negamax(position.clone(), alpha, beta, depth);

                // If failing high/low, increase the aspiration window
                // and re-search the position
                if score <= alpha {
                    //  -INF <   [score]  <=  alpha  <=   beta    <= INF
                    //   |----------|----------|----------|----------|
                    //           ↑                   ↑ new beta = (alpha + beta) / 2
                    //           ↑ new alpha = (score - delta)
                    beta = (alpha + beta) / 2.0;
                    alpha = (score - delta).max(-INFINITE);
                } else if score >= beta {
                    //  -INF <    alpha  <=  beta  <=  [score]   <= INF
                    //   |----------|----------|----------|----------|
                    //                                         ↑ new beta = (score + delta)
                    beta = (score + delta).min(INFINITE);
                } else {
                    eprintln!(
                        "[WITHIN] score: {} alpha: {} beta: {} delta: {}",
                        score, alpha, beta, delta
                    );
                    // score is within the window
                    break;
                }

                eprintln!(
                    "[OUTSIDE] score: {} alpha: {} beta: {} delta: {}",
                    score, alpha, beta, delta
                );

                // increase delta exponentially
                delta *= 2.0;

                assert!(alpha >= -INFINITE && beta <= INFINITE);

                if self.aborted {
                    break;
                }
            }

            if self.aborted {
                break;
            }

            best_move = self.pv.get_mainline().first().cloned();
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

        best_move
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
        moves.sort_unstable_by_key(|mv| mv != self.pv.get_best_move(self.ply));

        assert!(moves.len() > 0, "at least one legal move");

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
                score = -self.negamax(chess_moved, -beta, -alpha, depth - 1)
            };

            self.ply -= 1;

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

    fn evaluate(&mut self, position: &Chess) -> f32 {
        // increase number of evals computed
        self.evals += 1;

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
