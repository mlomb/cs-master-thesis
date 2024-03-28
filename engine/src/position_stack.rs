use crate::defs::MAX_PLY;
use nn::nnue::{accumulator::NnueAccumulator, model::NnueModel};
use shakmaty::{Chess, Color, Move, Position};
use std::{cell::RefCell, rc::Rc};

pub type HashKey = shakmaty::zobrist::Zobrist32;

pub struct State {
    /// The current position
    /// We use the shakmaty library that is copy-make
    /// This could be replaced by make-unmake
    pos: Chess,

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
        self.nnue_accum.copy_from(&other.nnue_accum);
    }
}

impl PositionStack {
    pub fn new(nnue_model: Rc<RefCell<NnueModel>>) -> Self {
        PositionStack {
            index: 0,
            stack: std::array::from_fn(|_| State {
                pos: Chess::default(),
                nnue_accum: NnueAccumulator::new(nnue_model.clone()),
            }),
        }
    }

    /// Clears the stack and sets the position
    pub fn reset(&mut self, pos: &Chess) {
        self.index = 0;
        self.stack[0].pos = pos.clone();
        self.stack[0].nnue_accum.refresh(pos.board(), Color::White);
        self.stack[0].nnue_accum.refresh(pos.board(), Color::Black);
    }

    /// Get current chess position
    pub fn get(&self) -> &Chess {
        &self.stack[self.index].pos
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

        if let Some(mov) = mov {
            // update the NNUE accumulator (before making the move!)
            next_state
                .nnue_accum
                .update(&next_state.pos.board(), &mov, Color::White);
            next_state
                .nnue_accum
                .update(&next_state.pos.board(), &mov, Color::Black);

            // make a regular move
            next_state.pos.play_unchecked(&mov);
        } else {
            // make a null move
            // (forfeit the move and let the opponent play)
            next_state.pos = next_state.pos.clone().swap_turn().unwrap();
        }
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
