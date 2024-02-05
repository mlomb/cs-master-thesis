use crate::{
    encoding::encode_board,
    pv::PVTable,
    tt::{TFlag, TTable},
};
use ndarray::Array2;
use ort::{inputs, Session};
use shakmaty::{CastlingMode, Chess, Color, Move, MoveList, Position};
use std::{
    ops::Index,
    time::{Duration, Instant},
};

type Value = i32;

const INFINITE: Value = 50_000;

pub struct Search {
    /// ML model to evaluate positions
    model: Session,

    /// Number of nodes searched
    nodes: usize,
    /// Number of evals computed
    evals: usize,

    /// Current ply
    ply: usize,
    /// Principal variation table
    pv: PVTable,
    /// Transposition table
    tt: TTable,

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
            tt: TTable::new(1_000_000 * 20), // ~480MB
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

        let mut best_line = None;

        for depth in 1..=max_depth.unwrap_or(60) {
            let score = self.negamax(position.clone(), -INFINITE, INFINITE, depth, false);

            if self.aborted {
                // do not replace best line since the search is incomplete
                break;
            }

            best_line = Some(self.pv.get_mainline());

            eprint!(
                "info depth {} nodes {} evals {} time {} score cp {} pv ",
                depth,
                self.nodes,
                self.evals,
                self.start_time.elapsed().as_millis(),
                score
            );
            for mv in best_line.as_ref().unwrap() {
                eprint!("{} ", mv.to_uci(CastlingMode::Standard).to_string());
            }
            eprint!("\n");
        }

        best_line.unwrap().first().cloned()
    }

    fn quiescence(&mut self, position: Chess, mut alpha: Value, beta: Value) -> Value {
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

    fn negamax(
        &mut self,
        position: Chess,
        mut alpha: Value,
        beta: Value,
        depth: i32,
        allow_null: bool,
    ) -> Value {
        assert!(-INFINITE <= alpha && alpha < beta && beta <= INFINITE);
        assert!(depth >= 0);

        self.pv.reset(self.ply);

        // time control
        self.checkup();

        if self.aborted {
            // abort search (most likely time limit)
            return 0;
        }

        let hash_key = self.tt.hash(&position);
        let mut pv_move = None;

        if let Some(score) = self.tt.probe(hash_key, alpha, beta, depth, &mut pv_move) {
            // hit!
            return score;
        }

        // TODO: optimize
        if position.is_stalemate() {
            return 0;
        }

        if position.is_checkmate() {
            return -10_000;
        }

        if depth == 0 {
            // escape from recursion
            // run quiescence search
            //return self.quiescence(position, alpha, beta);
            self.nodes += 1;
            return self.evaluate(&position);
        }

        let in_check = position.is_check();
        let mut tt_alpha_flag = TFlag::Alpha;

        // Null Move Pruning
        // https://www.chessprogramming.org/Null_Move_Pruning
        // TODO: watch out for more Zugzwang positions
        //
        // Before trying a null move, make sure that:
        // - we are allowed to make a null move (avoid twice in a row)
        // - we are not in the first ply
        // - we have remaining depth
        // - not in check
        // - TODO: at least one major piece still on the board
        //
        // R: depth reduction value
        // https://www.chessprogramming.org/Depth_Reduction_R
        const R: i32 = 2;
        if allow_null && self.ply > 0 && depth >= R + 1 && !in_check {
            // make a null move
            // (forfeit the move and let the opponent play)
            let null_position = unsafe { position.clone().swap_turn().unwrap_unchecked() };

            let score = -self.negamax(null_position, -beta, -beta + 1, depth - R - 1, false);

            if score >= beta {
                return beta;
            }
        }

        // increment the number of nodes searched
        self.nodes += 1;

        // generate legal moves
        let mut moves = position.legal_moves();
        assert!(moves.len() > 0, "at least one legal move");

        // sort moves
        self.sort_moves(&mut moves, pv_move);

        let mut best_move = None;
        let mut best_score = -INFINITE;

        for move_ in moves {
            let mut chess_moved = position.clone();
            chess_moved.play_unchecked(&move_);

            self.ply += 1;
            let score = -self.negamax(chess_moved, -beta, -alpha, depth - 1, true);
            self.ply -= 1;

            // replace best
            if score > best_score {
                best_score = score;
                best_move = Some(move_.clone());
            }

            // fail-hard beta cutoff
            if score >= beta {
                // store
                self.tt.record(hash_key, move_, beta, depth, TFlag::Beta);

                // TODO: killer moves

                // fails high
                return beta;
            }

            // found a better move!
            if score > alpha {
                tt_alpha_flag = TFlag::Exact;

                // PV node
                alpha = score;

                // write the move into the PV table
                self.pv.write(self.ply, move_);
            }
        }

        // TODO: checkmate?

        self.tt
            .record(hash_key, best_move.unwrap(), alpha, depth, tt_alpha_flag);

        // node fails low
        alpha
    }

    fn sort_moves(&self, moves: &mut MoveList, pv_move: Option<Move>) {
        // sort using the NN (too slow)
        // moves.sort_by_cached_key(|mv| {
        //     let mut chess_moved = position.clone();
        //     chess_moved.play_unchecked(&mv);
        //     -(self.evaluate(&chess_moved) * 100000.0) as i32
        // });

        let pv_move2 = self.pv.get_best_move(self.ply);

        // sort using an heuristic
        moves.sort_by_cached_key(|mv| {
            let mut score = 0;

            if mv == pv_move2 {
                score += 30_000;
            }

            // put the PV move first
            if let Some(pv_move) = &pv_move {
                if *mv == *pv_move {
                    score += 20_000;
                }
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

                score += MVV_LVA[attacker as usize][victim as usize] + 10_000;
            }

            //eprintln!(
            //    "ply={} mv={} truepv={} hashpv={:?} score={}",
            //    self.ply, mv, pv_move2, pv_move, score
            //);

            // sorts are from low to high, so flip
            -score
        });
    }

    fn evaluate(&mut self, position: &Chess) -> Value {
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

            return (value * 100.0) as Value;
        }
    }

    fn basic_eval(position: &Chess) -> Value {
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

        score
    }

    fn checkup(&mut self) {
        if self.nodes & 2047 == 0 {
            // make sure we are not exceeding the time limit
            if let Some(time_limit) = self.time_limit {
                if self.start_time.elapsed() >= time_limit {
                    self.aborted = true;
                }
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
