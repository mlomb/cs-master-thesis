use std::cmp::Ordering;
use std::ops::Neg;

pub trait Value {
    /// Custom comparison function for values
    fn compare(&self, other: &Self) -> Ordering;

    /// Negate to change the POV
    fn negate(&self) -> Self;
}

/// Implement Value for all types that implement Ord, Neg and Copy
impl<T: Ord + Neg<Output = T> + Copy> Value for T {
    fn compare(&self, other: &Self) -> Ordering {
        self.cmp(other)
    }

    fn negate(&self) -> Self {
        self.neg()
    }
}
