pub trait ValueComparator<Value> {
    /// Custom comparison function for values
    fn is_better(&self, candidate: &Value, actual_best: &Value) -> bool;
}

pub trait PositionEvaluator<Position, Value> {
    /// Evaluates a position into a Value
    fn eval(&self, state: &Position) -> Value;
}
