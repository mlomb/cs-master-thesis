use rand::seq::SliceRandom;

use crate::core::{outcome::Outcome, position};

pub trait Strategy<Position: position::Position> {
    /// --
    fn rollout(&self, position: &Position) -> f64;
}

#[derive(Clone)]
pub struct DefaultStrategy;

impl<Position> Strategy<Position> for DefaultStrategy
where
    Position: position::Position,
{
    fn rollout(&self, position: &Position) -> f64 {
        let mut pos = position.clone();
        let mut status = pos.status();
        let mut mult = -1.0;

        while let None = status {
            let actions = pos.valid_actions();
            let action = actions.choose(&mut rand::thread_rng()).unwrap();
            pos = pos.apply_action(action);
            status = pos.status();
            mult = -mult;
        }

        mult * match status {
            Some(Outcome::Win) => 1.0,
            Some(Outcome::Loss) => -1.0,
            Some(Outcome::Draw) => 0.0,
            None => unreachable!(),
        }
    }
}
