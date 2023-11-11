use std::{cmp::Ordering, ops::Neg};

pub trait Value: Clone {
    fn compare(&self, other: &Self) -> Ordering;
    fn reverse(&self) -> Self;
}

/// Implementation for all types all signed numeric types
impl<T> Value for T
where
    T: Clone + Ord + Neg<Output = Self>,
{
    fn compare(&self, other: &Self) -> Ordering {
        self.cmp(other)
    }

    fn reverse(&self) -> Self {
        self.clone().neg()
    }
}
