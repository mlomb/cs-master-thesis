use std::{cmp::Ordering, ops::Neg};

pub trait Value: Clone {
    fn compare(&self, other: &Self) -> Ordering;
    fn opposite(&self) -> Self;
}

/// Implementation for all types all signed numeric types
impl<T> Value for T
where
    T: Clone + Ord + Neg<Output = Self>,
{
    fn compare(&self, other: &Self) -> Ordering {
        self.cmp(other)
    }

    fn opposite(&self) -> Self {
        self.clone().neg()
    }
}
