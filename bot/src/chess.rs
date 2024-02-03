use shakmaty::Position;

use crate::position::{GamePosition, Outcome};

impl GamePosition for shakmaty::Chess {
    type Action = shakmaty::Move;

    fn initial() -> Self {
        shakmaty::Chess::default()
    }

    fn valid_actions(&self, actions: &mut Vec<Self::Action>) {
        self.legal_moves().into_iter().for_each(|m| actions.push(m));
    }

    fn apply_action(&self, action: &Self::Action) -> Self {
        let mut next = self.clone();
        next.play_unchecked(action);
        next
    }

    fn status(&self) -> Option<Outcome> {
        use shakmaty::Outcome::*;

        match self.outcome() {
            Some(outcome) => match outcome {
                Decisive { winner: player } => Some(if player == self.turn() {
                    Outcome::Win
                } else {
                    Outcome::Loss
                }),
                Draw => Some(Outcome::Draw),
            },
            None => None,
        }
    }
}
