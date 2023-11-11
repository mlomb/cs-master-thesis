pub trait PositionEvaluator<Position, Value> {
    /// Evaluates a position into a Value
    fn eval(&self, position: &Position) -> Value;
}
