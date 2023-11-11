use super::{
    outcome::Outcome,
    value::{self, Value},
};
use std::cmp::Ordering;

/// The result of a search
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum SearchResult<Value> {
    /// The search has reached a terminal state. The true outcome of the game is known
    Terminal(Outcome),
    /// The search has reached a non-terminal state
    NonTerminal(Value),
}

use Ordering::*;
use Outcome::*;
use SearchResult::*;

impl<V> Value for SearchResult<V>
where
    V: value::Value,
{
    /// Compare two search results
    /// It uses the value policy to compare non-terminal states
    fn compare(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Terminal(left), Terminal(right)) => left.cmp(right), // compare status
            (NonTerminal(left), NonTerminal(right)) => left.compare(right), // compare values

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
    fn reverse(&self) -> Self {
        match self {
            Terminal(Win) => Terminal(Loss),
            Terminal(Loss) => Terminal(Win),
            Terminal(Draw) => Terminal(Draw),
            NonTerminal(value) => NonTerminal(value.reverse()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orders() {
        // need to defined type for Value
        type R = SearchResult<i32>;

        // Win > eval > Draw > Loss
        assert_eq!(Greater, Terminal(Win).compare(&NonTerminal(1)));
        assert_eq!(Greater, NonTerminal(1).compare(&NonTerminal(-1)));
        assert_eq!(Greater, NonTerminal(-1).compare(&Terminal(Draw)));
        assert_eq!(Greater, R::Terminal(Draw).compare(&Terminal(Loss)));

        // Loss < Draw < eval < Win
        assert_eq!(Less, R::Terminal(Loss).compare(&Terminal(Draw)));
        assert_eq!(Less, Terminal(Draw).compare(&NonTerminal(-1)));
        assert_eq!(Less, NonTerminal(-1).compare(&NonTerminal(1)));
        assert_eq!(Less, NonTerminal(1).compare(&Terminal(Win)));
    }
}
