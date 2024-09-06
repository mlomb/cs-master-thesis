use crate::defs::{INVALID_MOVE, MAX_PLY};
use shakmaty::Move;

/// Triangular Principal Variation table
/// --------------------------------
/// PV line: e2e4 e7e5 g1f3 b8c6
pub struct PVTable {
    //      0    1    2    3    4    5
    // 0    m1   m2   m3   m4   m5   m6
    // 1    0    m2   m3   m4   m5   m6
    // 2    0    0    m3   m4   m5   m6
    // 3    0    0    0    m4   m5   m6
    // 4    0    0    0    0    m5   m6
    // 5    0    0    0    0    0    m6
    table: [[Move; MAX_PLY]; MAX_PLY],
    length: [usize; MAX_PLY],
}

impl PVTable {
    pub fn new() -> Self {
        PVTable {
            table: std::array::from_fn(|_| std::array::from_fn(|_| INVALID_MOVE)),
            length: [0; MAX_PLY],
        }
    }

    pub fn reset(&mut self, ply: usize) {
        assert!(ply < MAX_PLY - 1);
        self.length[ply] = ply;
    }

    pub fn write(&mut self, ply: usize, mov: Move) {
        // write new PV move
        self.table[ply][ply] = mov;

        for next_ply in ply + 1..self.length[ply + 1] {
            // copy from the deeper line
            self.table[ply][next_ply] = self.table[ply + 1][next_ply].clone();
        }

        // update length
        self.length[ply] = self.length[ply + 1];
    }

    /// Returns the main line of moves from the principal variation
    pub fn get_mainline(&self) -> Vec<Move> {
        (0..self.length[0])
            .map(|ply| self.table[0][ply].clone())
            .collect()
    }
}
