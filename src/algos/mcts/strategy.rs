use rand::seq::SliceRandom;

use crate::core::{outcome::Outcome, position};

pub trait Strategy<Action, Position>
where
    Action: Clone,
    Position: position::Position<Action>,
{
    /// --
    fn rollout(&self, position: &Position) -> f64;

    /// Defines the strategy for backpropagating the value
    fn backprop(&self, old_value: f64, new_value: f64) -> f64;
}

#[derive(Clone)]
pub struct DefaultStrategy;

impl<Action, Position> Strategy<Action, Position> for DefaultStrategy
where
    Action: Clone,
    Position: position::Position<Action>,
{
    fn rollout(&self, position: &Position) -> f64 {
        let mut pos = position.clone();
        let mut status = pos.status();

        while let None = status {
            let actions = pos.valid_actions();
            let action = actions.choose(&mut rand::thread_rng()).unwrap();
            pos = pos.apply_action(action);
            status = pos.status();
        }

        match status {
            Some(Outcome::Win) => 1.0,
            Some(Outcome::Loss) => -1.0,
            Some(Outcome::Draw) => 0.0,
            None => unreachable!(),
        }
    }

    fn backprop(&self, old_value: f64, new_value: f64) -> f64 {
        old_value + new_value
    }
}
