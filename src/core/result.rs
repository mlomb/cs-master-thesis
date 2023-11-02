use std::cmp::Ordering;
use std::ops::Neg;

use crate::core::outcome::*;
use crate::core::position::*;

#[derive(Debug)]
pub enum SearchResult<Value: Neg> {
    /// The game is over, and the true outcome is known
    True(Outcome),
    /// The game is not over, and the value is returned
    Eval(Value),
}

use Ordering::*;
use Outcome::*;
use SearchResult::*;

impl<Value: Ord + Neg> SearchResult<Value> {
    /// Custom comparison function for search results
    pub fn compare(&self, other: &Self) -> Ordering {
        match (self, other) {
            // Compare Status
            (True(left), True(right)) => left.cmp(right),
            // Compare Value
            (Eval(left), Eval(right)) => left.cmp(right),

            // Prefer true Win over Eval
            (True(Outcome::Win), Eval(_)) => Greater,
            (True(_), Eval(_)) => Less,

            // same as above
            (Eval(_), True(Outcome::Win)) => Less,
            (Eval(_), True(_)) => Greater,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orders() {
        type R = SearchResult<i32>;

        // WIN > eval > DRAW > LOSS
        assert_eq!(Equal, R::True(Win).compare(&R::True(Win)));
        assert_eq!(Greater, R::True(Win).compare(&R::Eval(-1)));
        assert_eq!(Greater, R::Eval(5).compare(&R::Eval(0)));
        assert_eq!(Greater, R::Eval(0).compare(&R::Eval(-5)));
        assert_eq!(Greater, R::Eval(-5).compare(&R::True(Draw)));
        assert_eq!(Greater, R::True(Draw).compare(&R::True(Loss)));
    }
}
