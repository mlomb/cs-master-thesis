use crate::core::outcome::*;
use crate::core::position::*;

/// Position for m-n-k games, such as TicTacToe and Connect4.
#[derive(Debug)]
pub struct MNK<const M: usize, const N: usize, const K: usize> {
    pub(crate) board: [u64; 2],
    pub(crate) who_plays: u8,
}

impl<const M: usize, const N: usize, const K: usize> MNK<M, N, K> {
    /// A bitboard with all the bits corresponding to the board set to 1.
    /// i.e 0b111_111_111 for 3x3
    const FULL_BOARD: u64 = (1 << (M * N)) - 1;

    // All bitboards that are winning sets for the game.
    // const WINNING_SETS: Vec<u64> = mnk_winning_sets(M, N, K);

    /// Returns a bitboard with all the used bits set to 1.
    pub fn occupied_board(&self) -> u64 {
        self.board[0] | self.board[1]
    }
}

impl<const M: usize, const N: usize, const K: usize> Position<usize> for MNK<M, N, K> {
    fn initial() -> Self {
        MNK {
            board: [0, 0],
            who_plays: 0,
        }
    }

    fn valid_actions(&self) -> Vec<usize> {
        let mut actions = Vec::new();
        let occupied = self.occupied_board();

        for i in 0..M * N {
            if (occupied & (1 << i)) == 0 {
                actions.push(i);
            }
        }

        actions
    }

    fn apply_action(&self, action: &usize) -> Self {
        let mut new_board = self.board;
        new_board[self.who_plays as usize] |= 1 << action;

        MNK {
            board: new_board,
            who_plays: 1 - self.who_plays,
        }
    }

    fn status(&self) -> Option<Outcome> {
        // TODO: make const or smth
        let winning_sets: Vec<u64> = mnk_winning_sets(M, N, K);

        for player in 0..2 {
            for &set in &winning_sets {
                if (self.board[player] as u64 & set) == set {
                    return if player as u8 == self.who_plays {
                        Some(Outcome::Win)
                    } else {
                        Some(Outcome::Loss)
                    };
                }
            }
        }

        if (self.board[0] | self.board[1]) == Self::FULL_BOARD {
            Some(Outcome::Draw)
        } else {
            None
        }
    }
}

pub type TicTacToe = MNK<3, 3, 3>;

/// Returns a list of all winning sets (as bitsets) for a game of size mxn with k in a row to win
pub(super) fn mnk_winning_sets(m: usize, n: usize, k: usize) -> Vec<u64> {
    assert!(m * n <= 64);
    assert!(k <= m && k <= n);

    let mut sets = Vec::new();

    // horizontal
    for i in 0..m {
        for j in 0..n - k + 1 {
            let mut set = 0;
            for l in 0..k {
                set |= 1 << (i * n + j + l);
            }

            sets.push(set);
        }
    }

    // vertical
    for i in 0..m - k + 1 {
        for j in 0..n {
            let mut set = 0;
            for l in 0..k {
                set |= 1 << ((i + l) * n + j);
            }

            sets.push(set);
        }
    }

    // diagonal
    for i in 0..m - k + 1 {
        for j in 0..n - k + 1 {
            let mut set = 0;
            for l in 0..k {
                set |= 1 << ((i + l) * n + j + l);
            }

            sets.push(set);
        }
    }

    // anti-diagonal
    for i in 0..m - k + 1 {
        for j in k - 1..n {
            let mut set = 0;
            for l in 0..k {
                set |= 1 << ((i + l) * n + j - l);
            }

            sets.push(set);
        }
    }

    sets
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn winning_set_tictactoe() {
        let ttt = mnk_winning_sets(3, 3, 3);
        assert_eq!(ttt.len(), 8);
        assert!(ttt.contains(&0b111_000_000));
        assert!(ttt.contains(&0b000_111_000));
        assert!(ttt.contains(&0b000_000_111));
        assert!(ttt.contains(&0b100_100_100));
        assert!(ttt.contains(&0b010_010_010));
        assert!(ttt.contains(&0b001_001_001));
        assert!(ttt.contains(&0b100_010_001));
        assert!(ttt.contains(&0b001_010_100));
    }

    #[test]
    fn winning_set_connect4() {
        let c4 = mnk_winning_sets(6, 7, 4);
        assert_eq!(c4.len(), 69);
    }
}
