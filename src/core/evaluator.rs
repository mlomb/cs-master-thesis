use crate::core::position;

use super::position::Outcome;

pub trait Evaluator<Action, Position: position::Position<Action>, Value> {
    /// Evaluates a position into a Value
    fn eval(&self, state: &Position) -> Value;
}
