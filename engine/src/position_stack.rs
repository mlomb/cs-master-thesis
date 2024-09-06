use crate::defs::{HashKey, MAX_PLY};
use nn::nnue::{accumulator::NnueAccumulator, model::NnueModel};
use shakmaty::{
    uci::UciMove,
    zobrist::{Zobrist32, ZobristHash},
    Chess, Color, EnPassantMode, Move, Position, Role,
};
use std::{cell::RefCell, rc::Rc};

pub struct State {
    /// The current position
    /// We use the shakmaty library that is copy-make
    /// This could be replaced by make-unmake, but it is fast enough (perf dominated by NNUE)
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

    /// Repetition list (previous to stack[0])
    repetitions: Vec<HashKey>,
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
            repetitions: Vec::with_capacity(128),
        }
    }

    /// Clears the stack and sets the position
    pub fn reset(&mut self, mut position: Chess, moves: Vec<UciMove>) {
        assert!(!position.legal_moves().is_empty(), "No moves available");

        self.index = 0;
        self.repetitions.clear();

        let mut rule50 = 0;

        // compute final position and state
        for mov in moves {
            let mov = mov.to_move(&position).unwrap();

            // increment rule 50
            if should_reset_50_rule(&mov) {
                rule50 = 0;
            } else {
                rule50 += 1;
            }

            // add previous move to repetition list
            self.repetitions
                .push(position.zobrist_hash(EnPassantMode::Legal));

            // advance position
            position = position.play(&mov).unwrap();
        }

        // fill first state
        self.stack[0].pos = position.clone();
        self.stack[0].hash_key = position.zobrist_hash(EnPassantMode::Legal);
        self.stack[0].rule50 = rule50;
        self.stack[0].nnue_accum.refresh(&position, Color::White);
        self.stack[0].nnue_accum.refresh(&position, Color::Black);
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
            // update the NNUE accumulator (BEFORE making the move!)
            next_state
                .nnue_accum
                .update(&next_state.pos, &mov, Color::White);
            next_state
                .nnue_accum
                .update(&next_state.pos, &mov, Color::Black);

            // make a regular move
            next_state.pos.play_unchecked(&mov);

            // reset rule50 counter on
            // - pawn moves
            // - captures
            if should_reset_50_rule(&mov) {
                next_state.rule50 = 0;
            }
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

    /// Returns true if the current position is a draw
    pub fn is_draw(&self) -> bool {
        let state = &self.stack[self.index];

        // insufficient material
        if state.pos.is_insufficient_material() {
            return true;
        }

        // 50-move rule
        if state.rule50 >= 100 {
            return true;
        }

        // threefold repetition
        let mut count = self
            .repetitions
            .iter()
            .filter(|&&x| x == state.hash_key)
            .count();
        for i in 0..self.index {
            if self.stack[i].hash_key == state.hash_key {
                count += 1;
            }
        }
        count >= 2
    }
}

/// Whether the 50-move rule should be reset after the move
fn should_reset_50_rule(mov: &Move) -> bool {
    // reset rule50 counter on
    // - captures
    // - pawn moves
    mov.is_capture() || mov.role() == Role::Pawn
}
