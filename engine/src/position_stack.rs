use crate::defs::MAX_PLY;
use nn::nnue::{accumulator::NnueAccumulator, model::NnueModel};
use shakmaty::{
    zobrist::{Zobrist32, ZobristHash},
    Chess, Color, EnPassantMode, Move, Position, Role,
};
use std::{cell::RefCell, rc::Rc};

pub type HashKey = shakmaty::zobrist::Zobrist32;

pub struct State {
    /// The current position
    /// We use the shakmaty library that is copy-make
    /// This could be replaced by make-unmake
    pos: Chess,

    /// The Zobrist hash key for the position
    hash_key: HashKey,

    /// Rule50 counter
    rule50: u32,

    // NNUE accumulator
    nnue_accum: NnueAccumulator,
}

pub struct PositionStack {
    index: usize,
    stack: [State; MAX_PLY],
}

impl State {
    fn copy_from(&mut self, other: &State) {
        self.pos = other.pos.clone();
        self.hash_key = other.hash_key;
        self.rule50 = other.rule50;
        self.nnue_accum.copy_from(&other.nnue_accum);
    }
}

impl PositionStack {
    pub fn new(nnue_model: Rc<RefCell<NnueModel>>) -> Self {
        PositionStack {
            index: 0,
            stack: std::array::from_fn(|_| State {
                pos: Chess::default(),
                hash_key: Chess::default().zobrist_hash(EnPassantMode::Legal),
                rule50: 0,
                nnue_accum: NnueAccumulator::new(nnue_model.clone()),
            }),
        }
    }

    /// Clears the stack and sets the position
    pub fn reset(&mut self, pos: &Chess) {
        self.index = 0;
        self.stack[0].pos = pos.clone();
        self.stack[0].hash_key = pos.zobrist_hash(EnPassantMode::Legal);
        self.stack[0].rule50 = 0;
        self.stack[0].nnue_accum.refresh(pos.board(), Color::White);
        self.stack[0].nnue_accum.refresh(pos.board(), Color::Black);
    }

    /// Get current chess position
    pub fn get(&self) -> &Chess {
        &self.stack[self.index].pos
    }

    pub fn hash_key(&self) -> Zobrist32 {
        self.stack[self.index].hash_key
    }

    pub fn rule50(&self) -> u32 {
        self.stack[self.index].rule50
    }

    /// Makes a move, or a null move if None.
    /// The move is assumed to be legal.
    pub fn do_move(&mut self, mov: Option<Move>) {
        // increment stack and copy
        let (prevs, nexts) = self.stack.split_at_mut(self.index + 1);
        let prev_state = &prevs[self.index];
        let next_state = &mut nexts[0];
        self.index += 1;

        next_state.copy_from(prev_state);

        // increment rule50 counter
        // may be reset by a pawn move or a capture
        next_state.rule50 += 1;

        if let Some(mov) = mov {
            // make a regular move
            next_state.pos.play_unchecked(&mov);

            // reset rule50 counter on
            // - pawn moves
            // - captures
            if let Move::Normal { role, .. } = mov {
                if role == Role::Pawn || mov.is_capture() {
                    next_state.rule50 = 0;
                }
            }

            // update the NNUE accumulator (AFTER making the move!)
            next_state
                .nnue_accum
                .update(&next_state.pos.board(), Color::White);
            next_state
                .nnue_accum
                .update(&next_state.pos.board(), Color::Black);
        } else {
            // make a null move
            // (forfeit the move and let the opponent play)
            next_state.pos = next_state.pos.clone().swap_turn().unwrap();
        }

        next_state.hash_key = next_state.pos.zobrist_hash(EnPassantMode::Legal);
    }

    /// Undoes the last move
    pub fn undo_move(&mut self) {
        // decrement stack
        self.index -= 1;
    }

    /// Return the evaluation for the current position
    pub fn evaluate(&mut self) -> i32 {
        let side_to_move = self.get().turn();

        self.stack[self.index].nnue_accum.forward(side_to_move)
    }
}
