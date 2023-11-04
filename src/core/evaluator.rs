pub trait PositionEvaluator<Position, Value> {
    /// Evaluates a position into a Value
    fn eval(&self, state: &Position) -> Value;
}
