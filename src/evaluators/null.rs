use crate::core::evaluator::Evaluator;
use crate::core::position::Position;
use crate::core::value::Value;

pub struct NullEvaluator;

impl<A, P, V> Evaluator<A, P, V> for NullEvaluator
where
    A: Clone,
    P: Position<A>,
    V: Value,
{
    fn eval(&self, _: &P) -> V {
        panic!("NullEvaluator should never evaluate a position")
    }
}
