use super::{
    outcome::Outcome,
    value::{self, ValuePolicy},
};
use std::cmp::Ordering;

/// The result of a search
#[derive(Debug, PartialEq, Eq)]
pub enum SearchResult<Value> {
    /// The search has reached a terminal state. The true outcome of the game is known
    Terminal(Outcome),
    /// The search has reached a non-terminal state
    NonTerminal(Value),
}

use Ordering::*;
use Outcome::*;
use SearchResult::*;

impl<Value> SearchResult<Value> {
    /// Compare two search results
    /// It uses the value policy to compare non-terminal states
    pub fn compare(&self, other: &Self, value_policy: &mut dyn ValuePolicy<Value>) -> Ordering {
        match (self, other) {
            (Terminal(left), Terminal(right)) => left.cmp(right), // compare status
            (NonTerminal(left), NonTerminal(right)) => value_policy.compare(left, right), // compare values

            // Prefer true Win over eval
            (Terminal(Win), NonTerminal(_)) => Greater,
            (Terminal(_), NonTerminal(_)) => Less,

            // same as above
            (NonTerminal(_), Terminal(Win)) => Less,
            (NonTerminal(_), Terminal(_)) => Greater,
        }
    }

    /// Computes the opposite of a search result
    /// It uses the value policy to compute the opposite of non-terminal states
    pub fn opposite(&self, value_policy: &mut dyn ValuePolicy<Value>) -> Self {
        match self {
            Terminal(Win) => Terminal(Loss),
            Terminal(Loss) => Terminal(Win),
            Terminal(Draw) => Terminal(Draw),
            NonTerminal(value) => NonTerminal(value_policy.opposite(value)),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::core::value::DefaultValuePolicy;

    use super::*;

    #[test]
    fn orders() {
        type R = SearchResult<i32>;
        let mut p = DefaultValuePolicy;

        // Win > eval > Draw > Loss
        assert_eq!(Greater, R::Terminal(Win).compare(&NonTerminal(1), &mut p));
        assert_eq!(Greater, R::NonTerminal(1).compare(&NonTerminal(-1), &mut p));
        assert_eq!(Greater, R::NonTerminal(-1).compare(&Terminal(Draw), &mut p));
        assert_eq!(Greater, R::Terminal(Draw).compare(&Terminal(Loss), &mut p));
    }
}
