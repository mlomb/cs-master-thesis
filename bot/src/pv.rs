use shakmaty::Move;
use std::mem::MaybeUninit;

const MAX_PLY: usize = 64;

/// Triangular PV table
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
    table: [[MaybeUninit<Move>; MAX_PLY]; MAX_PLY],
    length: [usize; MAX_PLY],
}

impl PVTable {
    pub fn new() -> Self {
        PVTable {
            table: std::array::from_fn(|_| std::array::from_fn(|_| MaybeUninit::uninit())),
            length: [0; MAX_PLY],
        }
    }

    pub fn reset(&mut self, ply: usize) {
        assert!(ply < MAX_PLY - 1);
        self.length[ply] = ply;
    }

    pub fn write(&mut self, ply: usize, move_: Move) {
        // write new PV move
        self.table[ply][ply] = MaybeUninit::new(move_);

        for next_ply in ply + 1..self.length[ply + 1] {
            // copy from the deeper line
            self.table[ply][next_ply] =
                MaybeUninit::new(unsafe { self.table[ply + 1][next_ply].assume_init_read() });
        }

        // update length
        self.length[ply] = self.length[ply + 1];
    }

    pub fn get_best_move(&self, ply: usize) -> &Move {
        unsafe { self.table[ply][ply].assume_init_ref() }
    }

    pub fn get_mainline(&self) -> Vec<Move> {
        (0..self.length[0])
            .map(|ply| unsafe { self.table[0][ply].assume_init_read() })
            .collect()
    }
}
