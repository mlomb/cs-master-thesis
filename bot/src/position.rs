use crate::defs::MAX_PLY;
use shakmaty::{Board, Chess, Move, MoveList};

pub type HashKey = shakmaty::zobrist::Zobrist64;

pub struct Position {
    /// Stack of chess positions
    /// We use the shakmaty library that is copy-make
    /// This could be replaced by make-unmake
    stack: Vec<Chess>,
}

impl Position {
    /// Creates a Position from a shakmaty Chess
    pub fn from_chess(chess: Chess) -> Self {
        let mut stack = Vec::with_capacity(MAX_PLY);
        stack.push(chess);
        Position { stack }
    }

    /// Creates a Position in the starting position
    pub fn start_pos() -> Self {
        Position::from_chess(Chess::default())
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
    pub fn undo_move(&mut self, _m: Option<Move>) {
        self.stack.pop();
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
