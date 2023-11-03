use crate::core::evaluator::Evaluator;
use crate::core::position::Position;
use crate::core::value::Value;

pub struct NullEvaluator;

impl<P, V> Evaluator<P, V> for NullEvaluator
where
    P: Position,
    V: Value,
{
    fn eval(&self, _: &P) -> V {
        panic!("NullEvaluator should never evaluate a position")
    }
}
