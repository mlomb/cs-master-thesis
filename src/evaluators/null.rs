use crate::core::evaluator::{PositionEvaluator, ValueComparator};
use crate::core::position::Position;

pub struct NullEvaluator;

impl<V: Ord> ValueComparator<V> for NullEvaluator {
    fn is_better(&self, _: &V, _: &V) -> bool {
        panic!("NullEvaluator should never compare values")
    }
}

impl<P, V: Ord> PositionEvaluator<P, V> for NullEvaluator
where
    P: Position,
{
    fn eval(&self, _: &P) -> V {
        panic!("NullEvaluator should never evaluate a position")
    }
}
