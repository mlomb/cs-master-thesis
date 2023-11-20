use super::position;
use rand::seq::SliceRandom;

pub trait Agent<Position>
where
    Position: position::Position,
{
    /// Returns the next action to take given the current state.
    fn next_action(&mut self, position: &Position) -> Option<Position::Action>;
}

#[derive(Clone)]
pub struct RandomAgent {}

impl<Position> Agent<Position> for RandomAgent
where
    Position: position::Position,
{
    fn next_action(&mut self, position: &Position) -> Option<Position::Action> {
        position
            .valid_actions()
            .choose(&mut rand::thread_rng())
            .cloned()
    }
}
