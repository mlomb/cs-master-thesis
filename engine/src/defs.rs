use shakmaty::{Move, Role, Square};

/// Engine's score units
pub type Value = i32;

/// Infinity for a Value
pub const INFINITE: Value = 50_000;

/// Maximum number of plies the engine supports
pub const MAX_PLY: usize = 64;

/// Invalid move for initialization
pub const INVALID_MOVE: Move = Move::Normal {
    role: Role::Pawn,
    from: Square::A1,
    to: Square::A1,
    capture: None,
    promotion: None,
};
