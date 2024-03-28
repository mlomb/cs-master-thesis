use crate::position_stack::HashKey;
use shakmaty::Move;

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
pub struct TTable {
    entries: Vec<TEntry>,
}

impl TTable {
    pub fn new(size: usize) -> Self {
        TTable {
            entries: vec![
                TEntry {
                    key: HashKey::default(),
                    flag: TFlag::Exact,
                    depth: -1,
                    score: 0,
                    move_: None,
                };
                size
            ],
        }
    }

    pub fn record(&mut self, key: HashKey, move_: Move, score: i32, depth: i32, flag: TFlag) {
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

    pub fn probe(
        &self,
        key: HashKey,
        alpha: i32,
        beta: i32,
        depth: i32,
        pv_move: &mut Option<Move>,
    ) -> Option<i32> {
        let entry = &self.entries[key.0 as usize % self.entries.len()];

        // make sure the position is the same (note that there can still be collisions)
        if entry.key == key {
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

        None
    }

    /*

    pub fn probe_pv_line(&self, position: &Chess, mut max_depth: i32) -> Vec<Move> {
        let mut line = vec![];
        let mut position = position.clone();

        while let Some(move_) = self.probe_pv_move(&position) {
            if max_depth == 0 {
                break;
            }
            max_depth -= 1;

            if position.is_legal(&move_) {
                position.play_unchecked(&move_);
                line.push(move_);
            } else {
                break;
            }
        }

        line
    }

    pub fn probe_pv_move(&self, position: &Chess) -> Option<Move> {
        let key = self.hash(position);
        let entry = &self.entries[key.0 as usize % self.entries.len()];

        if entry.key == key {
            return entry.move_.clone();
        }

        None
    }

    */
}
