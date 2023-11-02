use std::cmp::Ordering;

pub trait Value {
    /// Custom comparison function for values
    fn compare(&self, other: &Self) -> Ordering;
}

/// Implement Value for all types that implement Ord
impl<T: Ord> Value for T {
    fn compare(&self, other: &Self) -> Ordering {
        self.cmp(other)
    }
}
