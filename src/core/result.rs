use std::cmp::Ordering;
use std::ops::Neg;

use crate::core::position::*;

pub enum SearchResult<Value: Neg> {
    Outcome(Outcome),
    Eval(Value),
}

use Ordering::*;
use Outcome::*;
use SearchResult::*;

impl<Value: Ord + Neg> SearchResult<Value> {
    /// Custom comparison function for search results
    pub fn compare(&self, other: &Self) -> Ordering {
        match (self, other) {
            // Directly compare Status
            (Outcome(left), Outcome(right)) => left.cmp(right),
            // Directly compare Value
            (Eval(left), Eval(right)) => left.cmp(right),

            (Outcome(status), Eval(_)) => {
                if status >= &PLAYING {
                    Greater
                } else {
                    Less
                }
            }
            (Eval(_), Outcome(status)) => {
                if status >= &PLAYING {
                    Less
                } else {
                    Greater
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orders() {
        type R = SearchResult<i32>;

        // WIN > PLAYING > eval > DRAW > LOSS
        assert_eq!(Equal, R::Outcome(WIN).compare(&R::Outcome(WIN)));
        assert_eq!(Greater, R::Outcome(WIN).compare(&R::Outcome(PLAYING)));
        assert_eq!(Greater, R::Outcome(PLAYING).compare(&R::Eval(-1)));
        assert_eq!(Greater, R::Eval(5).compare(&R::Eval(0)));
        assert_eq!(Greater, R::Eval(0).compare(&R::Eval(-5)));
        assert_eq!(Greater, R::Eval(-5).compare(&R::Outcome(DRAW)));
        assert_eq!(Greater, R::Outcome(DRAW).compare(&R::Outcome(LOSS)));
    }
}
