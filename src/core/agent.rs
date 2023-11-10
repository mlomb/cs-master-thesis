use super::position;

pub trait Agent<Position>
where
    Position: position::Position,
{
    /// Returns the next action to take given the current state.
    fn next_action(&mut self, position: &Position) -> Option<Position::Action>;
}
