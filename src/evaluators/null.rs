use crate::core::evaluator::PositionEvaluator;
use crate::core::position::Position;

pub struct NullEvaluator;

impl<P, V: Ord> PositionEvaluator<P, V> for NullEvaluator
where
    P: Position,
{
    fn eval(&self, _: &P) -> V {
        panic!("NullEvaluator should never evaluate a position")
    }
}
