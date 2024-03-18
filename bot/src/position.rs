use crate::defs::MAX_PLY;
use nn::nnue::model::NnueModel;
use shakmaty::{Board, Chess, Color, Move, MoveList};
use std::{cell::RefCell, rc::Rc};

pub type HashKey = shakmaty::zobrist::Zobrist64;

pub struct Position {
    /// Stack of chess positions
    /// We use the shakmaty library that is copy-make
    /// This could be replaced by make-unmake
    stack: Vec<Chess>,

    /// NNUE
    features_buffer: Vec<u16>,
    added_features: Vec<u16>,
    removed_features: Vec<u16>,
    nnue_model: Rc<RefCell<NnueModel>>,
}

impl Position {
    /// Creates a Position from a shakmaty Chess
    pub fn from_chess(chess: Chess, nnue_model: Rc<RefCell<NnueModel>>) -> Self {
        let mut stack = Vec::with_capacity(MAX_PLY);
        stack.push(chess);

        Position {
            stack,
            added_features: Vec::with_capacity(768),
            removed_features: Vec::with_capacity(768),
            features_buffer: Vec::with_capacity(768),
            nnue_model,
        }
    }

    pub fn refresh(&mut self) {
        use shakmaty::Position;

        let pos = self.current().clone();

        for &perspective in [Color::White, Color::Black].iter() {
            self.features_buffer.clear();

            self.nnue_model.borrow().get_feature_set().active_features(
                pos.board(),
                perspective,
                &mut self.features_buffer,
            );
            self.nnue_model
                .borrow_mut()
                .refresh_accumulator(&self.features_buffer, perspective);
        }
    }

    fn current(&self) -> &Chess {
        self.stack.last().unwrap()
    }

    pub fn board(&self) -> Board {
        use shakmaty::Position;
        self.current().board().clone()
    }

    /// Makes a move, or a null move if None.
    /// The move is assumed to be legal.
    pub fn do_move(&mut self, m: Option<Move>) {
        use shakmaty::Position;

        let mut new_pos = self.current().clone();

        if let Some(m) = m {
            // update the NNUE accumulator

            //println!("Do move: {:?}", m);

            for &perspective in [Color::White, Color::Black].iter() {
                self.added_features.clear();
                self.removed_features.clear();

                self.nnue_model.borrow().get_feature_set().changed_features(
                    new_pos.board(),
                    &m,
                    perspective,
                    &mut self.added_features,
                    &mut self.removed_features,
                );
                self.nnue_model.borrow_mut().update_accumulator(
                    &self.added_features,
                    &self.removed_features,
                    perspective,
                );
            }

            // make a regular move
            new_pos.play_unchecked(&m);
        } else {
            // make a null move
            // (forfeit the move and let the opponent play)
            new_pos = unsafe { new_pos.swap_turn().unwrap_unchecked() };
        }

        self.stack.push(new_pos);
    }

    /// Undoes the last move
    pub fn undo_move(&mut self, m: Option<Move>) {
        use shakmaty::Position;

        self.stack.pop().unwrap();
        let pos = self.current().clone();

        if let Some(m) = m {
            // update the NNUE accumulator
            for &perspective in [Color::White, Color::Black].iter() {
                self.added_features.clear();
                self.removed_features.clear();

                self.nnue_model.borrow().get_feature_set().changed_features(
                    pos.board(),
                    &m,
                    perspective,
                    &mut self.added_features,
                    &mut self.removed_features,
                );
                self.nnue_model.borrow_mut().update_accumulator(
                    &self.removed_features,
                    &self.added_features,
                    perspective,
                );
            }
        }
    }

    pub fn evaluate(&mut self) -> i32 {
        use shakmaty::Position;

        let pos = self.current().clone();
        let side_to_move = pos.turn(); // side to move

        let ue_value = self.nnue_model.borrow_mut().forward(side_to_move);

        //self.refresh();
        //let refresh_value = self.nnue_model.borrow_mut().forward(side_to_move);
        //if ue_value != refresh_value {
        //    println!("UE: {}, REFRESH: {}", ue_value, refresh_value);
        //    panic!();
        //}

        ue_value
    }

    /// Generates all legal moves
    pub fn legal_moves(&self) -> MoveList {
        use shakmaty::Position;
        // we don't cache the moves since it should be used at most once per position
        self.current().legal_moves()
    }

    pub fn compute_hash(&self) -> HashKey {
        use shakmaty::zobrist::ZobristHash;
        self.current().zobrist_hash(shakmaty::EnPassantMode::Legal)
    }

    pub fn is_check(&self) -> bool {
        use shakmaty::Position;
        self.current().is_check()
    }

    pub fn turn(&self) -> shakmaty::Color {
        use shakmaty::Position;
        self.current().turn()
    }
}
