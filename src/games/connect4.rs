use super::mnk::MNK;
use crate::core::outcome::Outcome;
use crate::core::position::Position;
use core::fmt;

const ROWS: usize = 6;
const COLS: usize = 7;

#[derive(Clone, Debug)]
pub struct Connect4(MNK<ROWS, COLS, 4>);

impl Position for Connect4 {
    type Action = usize;

    fn initial() -> Self {
        Connect4(MNK::initial())
    }

    fn valid_actions(&self) -> Vec<usize> {
        let mut actions = Vec::new();

        let occupied = self.0.occupied_board();
        for col in 0..COLS {
            if (occupied & (1 << col)) == 0 {
                actions.push(col);
            }
        }

        actions
    }

    fn apply_action(&self, action: &usize) -> Self {
        assert!(action < &COLS);

        let occupied = self.0.occupied_board();

        let mut row = 0;
        while row < ROWS - 1 {
            let next_bit = 1 << ((row + 1) * COLS + action);
            if (occupied & next_bit) != 0 {
                break;
            }
            row += 1;
        }

        let mut new_board = self.0.board;
        new_board[self.0.who_plays as usize] |= 1 << (row * COLS + action);

        Connect4(MNK {
            board: new_board,
            who_plays: 1 - self.0.who_plays,
        })
    }

    fn status(&self) -> Option<Outcome> {
        self.0.status()
    }
}

impl fmt::Display for Connect4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}
