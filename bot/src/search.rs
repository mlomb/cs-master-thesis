use crate::{
    defs::{Value, INFINITE, INVALID_MOVE, MAX_PLY},
    position::Position,
    pv::PVTable,
    tt::{TFlag, TTable},
};
use nn::nnue::NnueModel;
use shakmaty::{
    zobrist::{Zobrist64, ZobristHash},
    CastlingMode, Chess, Move, MoveList,
};
use std::time::{Duration, Instant};

pub struct Search {
    /// Current position
    pub pos: Position,
    /// Current ply
    pub ply: usize,

    /// Neural network model
    pub nnue_model: NnueModel,

    /// Number of nodes searched
    pub nodes: usize,
    /// Number of evals computed
    pub evals: usize,

    /// Principal variation table
    pub pv: PVTable,
    /// Transposition table
    pub tt: TTable,
    /// Killer moves
    pub killer_moves: [[Move; 2]; MAX_PLY],
    /// History moves
    pub history_moves: [[Value; 8 * 8]; 12],
    /// Repetition table
    pub repetition_index: usize,
    pub repetition_table: [Zobrist64; 1024],

    /// Start search time
    pub start_time: Instant,
    /// Time limit, do not exceed this time
    pub time_limit: Option<Duration>,
    /// Whether the search was aborted
    pub aborted: bool,
}

impl Search {
    pub fn new(nnue_model: NnueModel) -> Self {
        Search {
            pos: Position::start_pos(),
            ply: 0,
            nnue_model,
            nodes: 0,
            evals: 0,
            pv: PVTable::new(),
            tt: TTable::new(1_000_000 * 20), // ~480MB
            killer_moves: std::array::from_fn(|_| [INVALID_MOVE, INVALID_MOVE]),
            history_moves: [[0; 8 * 8]; 12],
            repetition_index: 0,
            repetition_table: [Zobrist64(0); 1024],
            start_time: Instant::now(),
            time_limit: None,
            aborted: false,
        }
    }

    pub fn go(
        &mut self,
        position: Chess,
        max_depth: Option<i32>,
        time_limit: Option<Duration>,
    ) -> Option<Move> {
        // init position
        self.pos = Position::from_chess(position);
        self.ply = 0;

        // reset stats
        self.nodes = 0;
        self.evals = 0;

        // time control
        self.start_time = Instant::now();
        self.time_limit = time_limit;
        self.aborted = false;

        // best line found so far
        let mut best_line = None;

        for depth in 1..=max_depth.unwrap_or(MAX_PLY as i32 - 1) {
            let score = self.negamax(-INFINITE, INFINITE, depth, false);

            if self.aborted {
                // time limit
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
                eprint!("{} ", mv.to_uci(CastlingMode::Standard));
            }
            eprint!("\n");
        }

        best_line.unwrap().first().cloned()
    }

    fn quiescence(&mut self, mut alpha: Value, beta: Value) -> Value {
        // increment the number of nodes searched
        self.nodes += 1;

        // evaluate the position
        let score = self.nnue_model.forward(self.pos.turn());
        // increase number of evals computed
        self.evals += 1;

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

        for move_ in self.pos.legal_moves() {
            if !move_.is_capture() {
                // only look at captures
                continue;
            }

            self.pos.do_move(Some(move_.clone()));
            self.ply += 1;
            let score = -self.quiescence(-beta, -alpha);
            self.ply -= 1;
            self.pos.undo_move(Some(move_));

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
        alpha
    }

    fn negamax(&mut self, mut alpha: Value, beta: Value, depth: i32, allow_null: bool) -> Value {
        assert!(-INFINITE <= alpha && alpha < beta && beta <= INFINITE);
        assert!(depth >= 0);

        // time control
        self.checkup();

        if self.aborted {
            // abort search (time limit)
            return 0;
        }

        self.pv.reset(self.ply);

        let is_pv = beta - alpha > 1;
        let hash_key = self.pos.compute_hash();
        let mut pv_move = None;

        if !is_pv {
            if let Some(score) = self.tt.probe(hash_key, alpha, beta, depth, &mut pv_move) {
                // hit!
                return score;
            }
        }

        // check three fold repetition
        if self.ply > 0 && self.is_repetition(hash_key) {
            // draw
            return 0;
        }

        if depth == 0 {
            // escape from recursion
            // run quiescence search
            return self.quiescence(alpha, beta);
        }

        let in_check = self.pos.is_check();
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
            self.pos.do_move(None);
            self.ply += 1;
            let score = -self.negamax(-beta, -beta + 1, depth - R - 1, false);
            self.ply -= 1;
            self.pos.undo_move(None);

            if score >= beta {
                return beta;
            }
        }

        // increment the number of nodes searched
        self.nodes += 1;

        // generate legal moves
        let mut moves = self.pos.legal_moves();

        // sort moves
        self.sort_moves(&mut moves, pv_move);

        let mut best_move = None;
        let mut best_score = -INFINITE;

        for move_ in moves {
            let is_quiet = !move_.is_capture();

            self.ply += 1;
            self.push_repetition_key(hash_key);
            self.pos.do_move(Some(move_.clone()));
            let score = -self.negamax(-beta, -alpha, depth - 1, true);
            self.pos.undo_move(Some(move_.clone()));
            self.pop_repetition_key();
            self.ply -= 1;

            // replace best
            if score > best_score {
                best_score = score;
                best_move = Some(move_.clone());
            }

            // fail-hard beta cutoff
            if score >= beta {
                if is_quiet {
                    // store killer moves
                    self.killer_moves[self.ply][1] = self.killer_moves[self.ply][0].clone();
                    self.killer_moves[self.ply][0] = move_.clone();
                }

                // store TT entry
                self.tt.record(hash_key, move_, beta, depth, TFlag::Beta);

                // fails high
                return beta;
            }

            // found a better move!
            if score > alpha {
                if is_quiet {
                    // store history moves
                    self.history_moves[move_.role() as usize][move_.to() as usize] += depth * depth;
                }

                tt_alpha_flag = TFlag::Exact;

                // PV node
                alpha = score;

                // write the move into the PV table
                self.pv.write(self.ply, move_);
            }
        }

        if best_move.is_none() {
            if in_check {
                // checkmate
                return -10_000;
            } else {
                // stalemate (draw)
                return 0;
            }
        }

        self.tt
            .record(hash_key, best_move.unwrap(), alpha, depth, tt_alpha_flag);

        // node fails low
        alpha
    }

    fn sort_moves(&self, moves: &mut MoveList, pv_move: Option<Move>) {
        moves.sort_by_cached_key(|move_| {
            let mut score = 0;

            // put the PV move first
            if let Some(pv_move) = &pv_move {
                if *move_ == *pv_move {
                    score += 20_000;
                }
            }

            if move_.is_capture() {
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

                let attacker = move_.role();
                let victim = move_.capture().unwrap();

                score += MVV_LVA[attacker as usize][victim as usize] + 10_000;
            } else {
                // move is quiet
                score += if self.killer_moves[self.ply][0] == *move_ {
                    9000
                } else if self.killer_moves[self.ply][1] == *move_ {
                    8000
                } else {
                    self.history_moves[move_.role() as usize][move_.to() as usize]
                }
            }

            // sorts are from low to high, so flip
            -score
        });
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

/// ==============================
///       Repetition table
/// ==============================
impl Search {
    /// Clears the repetition table
    /// Most likely called at the start of a new game
    pub fn reset_repetition(&mut self) {
        self.repetition_index = 0;
    }

    /// Record a position in the repetition table
    /// Called after Uci commands
    pub fn record_repetition(&mut self, position: &Chess) {
        let key = position.zobrist_hash(shakmaty::EnPassantMode::Legal);
        self.push_repetition_key(key);
    }

    /// Push a key in the repetition table
    /// Called when making a move
    fn push_repetition_key(&mut self, key: Zobrist64) {
        self.repetition_table[self.repetition_index] = key;
        self.repetition_index += 1;
    }

    /// Pop a key from the repetition table
    /// Called when undoing a move
    fn pop_repetition_key(&mut self) {
        self.repetition_index -= 1;
    }

    fn is_repetition(&self, key: Zobrist64) -> bool {
        self.repetition_table[..self.repetition_index].contains(&key)
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
