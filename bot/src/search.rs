use crate::{encoding::encode_board, pv::PVTable};
use ndarray::Array2;
use ort::{inputs, Session};
use shakmaty::{CastlingMode, Chess, Color, Position};
use std::ops::Index;

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
    follow_pv: bool,
    /// Principal variation
    pv: PVTable,
}

impl Search {
    pub fn new(model: Session) -> Self {
        Search {
            model,
            nodes: 0,
            evals: 0,
            ply: 0,
            follow_pv: false,
            pv: PVTable::new(),
        }
    }

    pub fn go(&mut self, position: &Chess, max_depth: i32) {
        self.nodes = 0;
        self.evals = 0;
        self.ply = 0;

        for depth in 1..=max_depth {
            // enable following the PV
            self.follow_pv = true;

            // find the best move within a given position
            let score = self.negamax(position.clone(), f32::NEG_INFINITY, f32::INFINITY, depth);

            eprint!(
                "info depth {} nodes {} evals {} score cp {} pv ",
                depth, self.nodes, self.evals, score
            );
            for mv in self.pv.get_mainline() {
                eprint!("{} ", mv.to_uci(CastlingMode::Standard).to_string());
            }
            eprint!("\n");
        }
    }

    pub fn quiescence(&mut self, position: Chess, mut alpha: f32, beta: f32) -> f32 {
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

    pub fn negamax(&mut self, position: Chess, mut alpha: f32, beta: f32, depth: i32) -> f32 {
        // init PV
        let mut found_pv = false;
        self.pv.reset(self.ply);

        // TODO: optimize
        if position.is_stalemate() {
            return 0.0;
        }

        if depth == 0 {
            // escape from recursion
            // run quiescence search
            return self.quiescence(position, alpha, beta);
        }

        // increment the number of nodes searched
        self.nodes += 1;

        // generate legal moves
        let mut moves = position.legal_moves();
        if self.follow_pv {
            self.follow_pv = false;

            moves.sort_unstable_by_key(|mv| mv != self.pv.get_best_move(self.ply));
        }

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

    pub fn evaluate(&mut self, position: &Chess) -> f32 {
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
}
