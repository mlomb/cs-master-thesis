use crate::defs::HashKey;
use shakmaty::{Chess, Move, Position};

#[derive(Clone)]
pub enum TFlag {
    Alpha,
    Beta,
    Exact,
}

#[derive(Clone)]
pub struct TEntry {
    pub key: HashKey,
    pub flag: TFlag,
    pub depth: i8,
    pub score: i32,
    pub move_: Option<Move>,
}

/// Transposition table
pub struct TranspositionTable {
    entries: Vec<TEntry>,
}

impl TranspositionTable {
    pub fn new(size_mb: usize) -> Self {
        TranspositionTable {
            entries: vec![
                TEntry {
                    key: HashKey::default(),
                    flag: TFlag::Exact,
                    depth: -1,
                    score: 0,
                    move_: None,
                };
                size_mb * 1024 * 1024 / std::mem::size_of::<TEntry>()
            ],
        }
    }

    pub fn write_entry(&mut self, key: HashKey, move_: Move, score: i32, depth: i32, flag: TFlag) {
        let len = self.entries.len();
        let entry = &mut self.entries[key.0 as usize % len];

        // TODO: check for depth and key, if it is worth to replace the entry

        *entry = TEntry {
            key,
            flag,
            depth: depth as i8,
            score,
            move_: Some(move_),
        };
    }

    pub fn read_entry(
        &self,
        pos: &Chess,
        key: HashKey,
        alpha: i32,
        beta: i32,
        depth: i32,
        pv_move: &mut Option<Move>,
    ) -> Option<i32> {
        let entry = &self.entries[key.0 as usize % self.entries.len()];

        // make sure the position is the same (note that there can still be collisions)
        if entry.key == key {
            if let Some(ref mov) = entry.move_ {
                // check legality
                if pos.is_legal(&mov) {
                    // make sure depth is the same or higher (otherwise information may be incorrect)
                    if entry.depth as i32 >= depth {
                        match entry.flag {
                            TFlag::Exact => return Some(entry.score),
                            TFlag::Alpha => {
                                if entry.score <= alpha {
                                    return Some(alpha);
                                }
                            }
                            TFlag::Beta => {
                                if entry.score >= beta {
                                    return Some(beta);
                                }
                            }
                        }
                    }

                    *pv_move = entry.move_.clone();
                }
            }
        }

        None
    }
}
