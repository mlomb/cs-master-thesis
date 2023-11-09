use std::cmp::Ordering;

/// Trait for types that form an "order" relationship that
/// does not necessarily satisfy reflexivity, transitivity or antisymmetry.
pub trait LooseOrd {
    fn loose_cmp(&self, other: &Self) -> Ordering;
}

/// Implement `LooseOrd` for all types that implement `Ord`.
impl<T> LooseOrd for T
where
    T: Ord,
{
    fn loose_cmp(&self, other: &Self) -> Ordering {
        self.cmp(other)
    }
}

pub trait LooseOrdComparator {
    fn compare(&self, left: &Self, right: &Self) -> Ordering;
}
