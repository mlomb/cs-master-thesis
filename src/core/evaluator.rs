use crate::core::position;
use crate::core::value;

pub trait Evaluator<Position: position::Position, Value: value::Value> {
    /// Evaluates a position into a Value
    fn eval(&self, state: &Position) -> Value;
}
