use crate::{
    defs::{Value, INFINITE, INVALID_MOVE, MAX_PLY},
    position_stack::{HashKey, PositionStack},
    pv::PVTable,
    tt::{TFlag, TTable},
};
use nn::nnue::model::NnueModel;
use shakmaty::{
    zobrist::ZobristHash, CastlingMode, Chess, EnPassantMode, Move, MoveList, Position,
};
use std::time::{Duration, Instant};
use std::{cell::RefCell, rc::Rc};

pub struct Search {
    /// Current position
    pub pos: PositionStack,
    /// Current ply
    pub ply: usize,

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
    pub repetition_table: [HashKey; 1024],

    /// Start search time
    pub start_time: Instant,
    /// Time limit, do not exceed this time
    pub time_limit: Option<Duration>,
    /// Whether the search was aborted
    pub aborted: bool,
}

impl Search {
    pub fn new(nnue_model: Rc<RefCell<NnueModel>>) -> Self {
        Search {
            pos: PositionStack::new(nnue_model.clone()),
            ply: 0,
            nodes: 0,
            evals: 0,
            pv: PVTable::new(),
            tt: TTable::new(1_000_000 * 20), // ~480MB
            killer_moves: std::array::from_fn(|_| [INVALID_MOVE, INVALID_MOVE]),
            history_moves: [[0; 8 * 8]; 12],
            repetition_index: 0,
            repetition_table: [HashKey::default(); 1024],
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
        self.pos.reset(&position);
        self.ply = 0;
        assert!(
            !self.pos.get().legal_moves().is_empty(),
            "No moves available"
        );

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

            print!(
                "info depth {} nodes {} evals {} time {} score cp {} pv ",
                depth,
                self.nodes,
                self.evals,
                self.start_time.elapsed().as_millis(),
                score
            );
            for mv in best_line.as_ref().unwrap() {
                print!("{} ", mv.to_uci(CastlingMode::Standard));
            }
            print!("\n");

            if score.abs() >= 9950 {
                // mate found
                print!("info string mate found, stopping search\n");
                break;
            }
        }

        best_line.expect("No PV line found").first().cloned()
    }

    fn quiescence(&mut self, mut alpha: Value, beta: Value) -> Value {
        // increment the number of nodes searched
        self.nodes += 1;

        // evaluate the position
        let score = self.pos.evaluate();
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

        // generate legal moves
        let mut moves = self.pos.get().capture_moves();

        // sort moves
        self.sort_moves(&mut moves, None);

        for move_ in moves {
            debug_assert!(move_.is_capture());

            self.pos.do_move(Some(move_.clone()));
            self.ply += 1;
            let score = -self.quiescence(-beta, -alpha);
            self.ply -= 1;
            self.pos.undo_move();

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
        let hash_key = self.pos.get().zobrist_hash(EnPassantMode::Legal);
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

        let in_check = self.pos.get().is_check();

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
            self.pos.undo_move();

            if score >= beta {
                return beta;
            }
        }

        // increment the number of nodes searched
        self.nodes += 1;

        // generate legal moves
        let mut moves = self.pos.get().legal_moves();

        // sort moves
        self.sort_moves(&mut moves, pv_move);

        let mut best_move = None;
        let mut best_score = -INFINITE;
        let mut tt_alpha_flag = TFlag::Alpha;
        let mut moves_searched = 0;

        for move_ in moves {
            let is_quiet = !move_.is_capture();
            let mut score;

            // make move
            self.ply += 1;
            self.push_repetition_key(hash_key);
            self.pos.do_move(Some(move_.clone()));

            if moves_searched == 0 {
                // full depth, this is the first move
                // probably the PV line
                score = -self.negamax(-beta, -alpha, depth - 1, true);
            } else {
                // Late move reductions (LMR)
                const LMR_REDUCTION: i32 = 3;
                if moves_searched >= 2 &&
                    depth >= LMR_REDUCTION &&
                    !in_check &&
                    // no capture
                    !move_.is_capture() &&
                    // no promotion
                    !move_.is_promotion()
                {
                    // search with reduced depth
                    score = -self.negamax(-(alpha + 1), -alpha, depth - LMR_REDUCTION, allow_null);
                } else {
                    // make sure a full search is done
                    score = alpha + 1;
                }

                // PVS search
                if score > alpha {
                    score = -self.negamax(-(alpha + 1), -alpha, depth - 1, true);

                    if (score > alpha) && (score < beta) {
                        // failed check :(
                        // re-search with full depth
                        score = -self.negamax(-beta, -alpha, depth - 1, true);
                    }
                }
            }

            // undo move
            self.pos.undo_move();
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
                    self.history_moves
                        [(self.pos.get().turn() as usize) * 6 + (move_.role() as usize - 1)]
                        [move_.to() as usize] += depth * depth;
                }

                tt_alpha_flag = TFlag::Exact;

                // PV node
                alpha = score;

                // write the move into the PV table
                self.pv.write(self.ply, move_);
            }

            moves_searched += 1;
        }

        if best_move.is_none() {
            if in_check {
                // checkmate
                return -10_000 + self.ply as i32;
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
                pub const MVV_LVA: [i32; 36] = [
                    15, 25, 35, 45, 55, 65, // Pawn
                    14, 24, 34, 44, 54, 64, // Knight
                    13, 23, 33, 43, 53, 63, // Bishop
                    12, 22, 32, 42, 52, 62, // Rook
                    11, 21, 31, 41, 51, 61, // Queen
                    10, 20, 30, 40, 50, 60, // King
                ];

                let attacker = move_.role() as usize - 1;
                let victim = move_.capture().unwrap() as usize - 1;

                score += MVV_LVA[attacker * 6 + victim] + 10_000;
            } else {
                // move is quiet
                score += if self.killer_moves[self.ply][0] == *move_ {
                    9000
                } else if self.killer_moves[self.ply][1] == *move_ {
                    8000
                } else {
                    self.history_moves
                        [(self.pos.get().turn() as usize) * 6 + (move_.role() as usize - 1)]
                        [move_.to() as usize]
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
    fn push_repetition_key(&mut self, key: HashKey) {
        self.repetition_table[self.repetition_index] = key;
        self.repetition_index += 1;
    }

    /// Pop a key from the repetition table
    /// Called when undoing a move
    fn pop_repetition_key(&mut self) {
        self.repetition_index -= 1;
    }

    fn is_repetition(&self, key: HashKey) -> bool {
        self.repetition_table[..self.repetition_index].contains(&key)
    }
}
