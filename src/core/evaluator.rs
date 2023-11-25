pub trait PositionEvaluator<Position, Value> {
    /// Evaluates a position into a Value
    fn eval(&self, position: &Position) -> Value;
}

pub struct NullEvaluator;

impl<P, V: Ord> PositionEvaluator<P, V> for NullEvaluator
where
    P: super::position::Position,
{
    fn eval(&self, _: &P) -> V {
        panic!("NullEvaluator should never evaluate a position")
    }
}
