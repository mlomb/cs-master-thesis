use crate::core::{outcome, position};
use shakmaty::Position;

impl position::Position for shakmaty::Chess {
    type Action = shakmaty::Move;

    fn initial() -> Self {
        shakmaty::Chess::default()
    }

    fn valid_actions(&self) -> Vec<Self::Action> {
        self.legal_moves().into_iter().collect()
    }

    fn apply_action(&self, action: &Self::Action) -> Self {
        let mut next = self.clone();
        next.play_unchecked(action);
        next
    }

    fn status(&self) -> Option<outcome::Outcome> {
        use shakmaty::Outcome::*;

        match self.outcome() {
            Some(outcome) => match outcome {
                Decisive { winner: player } => Some(if player == self.turn() {
                    outcome::Outcome::Win
                } else {
                    outcome::Outcome::Loss
                }),
                Draw => Some(outcome::Outcome::Draw),
            },
            None => None,
        }
    }
}
