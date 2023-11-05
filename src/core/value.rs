use std::{cmp::Ordering, ops::Neg};

pub trait ValuePolicy<Value> {
    fn compare(&mut self, left: &Value, right: &Value) -> Ordering;
    fn opposite(&mut self, value: &Value) -> Value;
}

pub struct DefaultValuePolicy;

/// Implementation for all types all signed numeric types
impl<V> ValuePolicy<V> for DefaultValuePolicy
where
    V: Copy + Ord + Neg<Output = V>,
{
    fn compare(&mut self, left: &V, right: &V) -> Ordering {
        left.cmp(right)
    }

    fn opposite(&mut self, value: &V) -> V {
        value.neg()
    }
}
