use crate::core::outcome::*;
use crate::core::value;
use std::cmp::Ordering;

#[derive(Debug, PartialEq)]
pub enum SearchResult<Value> {
    /// The game is over, and the true outcome is known
    True(Outcome),
    /// The game is not over, and the value is returned
    Eval(Value),
}

use Ordering::*;
use Outcome::*;
use SearchResult::*;

impl<Value: value::Value> value::Value for SearchResult<Value> {
    /// Custom comparison function for search results
    fn compare(&self, other: &Self) -> Ordering {
        match (self, other) {
            // Compare Status
            (True(left), True(right)) => left.cmp(right),
            // Compare Value
            (Eval(left), Eval(right)) => left.compare(right),

            // Prefer true Win over Eval
            (True(Win), Eval(_)) => Greater,
            (True(_), Eval(_)) => Less,

            // same as above
            (Eval(_), True(Win)) => Less,
            (Eval(_), True(_)) => Greater,
        }
    }

    fn negate(&self) -> Self {
        match self {
            True(outcome) => True(match outcome {
                Win => Loss,
                Draw => Draw,
                Loss => Win,
            }),
            Eval(value) => Eval(value.negate()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::value::Value;

    #[test]
    fn orders() {
        type R = SearchResult<i32>; // need a type for A

        // Win > Eval > Draw > Loss
        assert_eq!(Equal, R::True(Win).compare(&R::True(Win)));
        assert_eq!(Greater, R::True(Win).compare(&R::Eval(1)));
        assert_eq!(Greater, R::Eval(1).compare(&R::Eval(-1)));
        assert_eq!(Greater, R::Eval(-1).compare(&R::True(Draw)));
        assert_eq!(Greater, R::True(Draw).compare(&R::True(Loss)));
    }
}
