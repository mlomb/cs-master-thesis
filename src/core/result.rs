use crate::core::outcome::*;
use crate::core::value;
use std::cmp::Ordering;

#[derive(Debug)]
pub enum SearchResult<Value> {
    /// The game is over, and the true outcome is known
    True(Outcome),
    /// The game is not over, and the value is returned
    Eval(Value),
}

use Ordering::*;
use Outcome::*;
use SearchResult::*;

impl<Value: value::Value> SearchResult<Value> {
    /// Custom comparison function for search results
    pub fn compare(&self, other: &Self) -> Ordering {
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orders() {
        type R = SearchResult<i32>; // need a type for A

        // Win > Eval > Draw > Loss
        assert_eq!(Equal, R::True(Win).compare(&R::True(Win)));
        assert_eq!(Greater, R::True(Win).compare(&R::Eval(-1)));
        assert_eq!(Greater, R::Eval(5).compare(&R::Eval(0)));
        assert_eq!(Greater, R::Eval(0).compare(&R::Eval(-5)));
        assert_eq!(Greater, R::Eval(-5).compare(&R::True(Draw)));
        assert_eq!(Greater, R::True(Draw).compare(&R::True(Loss)));
    }
}
