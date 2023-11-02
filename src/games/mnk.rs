use super::mnk_sets::mnk_winning_sets;
use crate::core::outcome::*;
use crate::core::position::*;

/// Position for m-n-k games, such as TicTacToe and Connect4.
#[derive(Debug)]
pub struct MNK<const M: usize, const N: usize, const K: usize> {
    pub(crate) board: [u64; 2],
    pub(crate) who_plays: u8,
}

impl<const M: usize, const N: usize, const K: usize> MNK<M, N, K> {
    /// Creates a new position from a string of the form "XO.XO.XO."
    pub fn from_str(board_str: &str, who_plays: char) -> Self {
        let mut board = [0, 0];
        let mut who_plays = match who_plays {
            'X' => 0,
            'O' => 1,
            _ => panic!("Invalid character for who_plays"),
        };

        for (i, c) in board_str.chars().enumerate() {
            match c {
                'X' => board[0] |= 1 << i,
                'O' => board[1] |= 1 << i,
                '.' => (),
                _ => panic!("Invalid character in board string"),
            }

            if c != '.' {
                who_plays = 1 - who_plays;
            }
        }

        MNK { board, who_plays }
    }

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
