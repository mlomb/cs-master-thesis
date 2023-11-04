use crate::core::outcome::*;

#[derive(Debug, PartialEq)]
pub enum SearchResult<Value> {
    /// The game is over, and the true outcome is known
    True(Outcome),
    /// The game is not over, and the value is returned
    Eval(Value),
}

use Outcome::*;
use SearchResult::*;

use super::evaluator::ValueComparator;

impl<Value> SearchResult<Value> {
    /// Returns whether this SearchResult is better than the other
    /// If both are Eval, they are compared with the provided function
    pub fn is_better_than(
        &self,
        other: &Self,
        value_comparator: &dyn ValueComparator<Value>,
    ) -> bool {
        match (self, other) {
            (True(left), True(right)) => left > right, // compare status
            (Eval(left), Eval(right)) => value_comparator.is_better(left, right), // compare value

            // Prefer true Win over Eval
            (True(Win), Eval(_)) => true,
            (True(_), Eval(_)) => false,

            // same as above
            (Eval(_), True(Win)) => false,
            (Eval(_), True(_)) => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orders() {
        struct DefaultComparator;

        impl ValueComparator<i32> for DefaultComparator {
            fn is_better(&self, candidate: &i32, actual_best: &i32) -> bool {
                candidate > actual_best
            }
        }

        // Win > Eval > Draw > Loss
        assert!(True(Win).is_better_than(&Eval(1), &DefaultComparator));
        assert!(Eval(1).is_better_than(&Eval(-1), &DefaultComparator));
        assert!(Eval(-1).is_better_than(&True(Draw), &DefaultComparator));
        assert!(True(Draw).is_better_than(&True(Loss), &DefaultComparator));
    }
}
