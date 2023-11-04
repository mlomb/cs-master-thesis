use super::outcome::Outcome;
use std::{cmp::Ordering, ops::Neg};

/// The result of a search
#[derive(Debug, PartialEq, Eq)]
pub enum SearchResult<Value>
where
    Value: PartialOrd + Neg<Output = Value>,
{
    /// The search has reached a terminal state. The true outcome of the game is known
    Terminal(Outcome),
    /// The search has reached a non-terminal state
    NonTerminal(Value),
}

use Ordering::*;
use Outcome::*;
use SearchResult::*;

impl<Value> PartialOrd for SearchResult<Value>
where
    Value: PartialOrd + Neg<Output = Value>,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            (Terminal(left), Terminal(right)) => left.partial_cmp(right), // compare status
            (NonTerminal(left), NonTerminal(right)) => left.partial_cmp(right), // compare value

            // Prefer true Win over eval
            (Terminal(Win), NonTerminal(_)) => Some(Greater),
            (Terminal(_), NonTerminal(_)) => Some(Less),

            // same as above
            (NonTerminal(_), Terminal(Win)) => Some(Less),
            (NonTerminal(_), Terminal(_)) => Some(Greater),
        }
    }
}

impl<Value> Neg for SearchResult<Value>
where
    Value: PartialOrd + Neg<Output = Value>,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        match self {
            Terminal(Win) => Terminal(Loss),
            Terminal(Loss) => Terminal(Win),
            Terminal(Draw) => Terminal(Draw),
            NonTerminal(value) => NonTerminal(-value),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orders() {
        type R = SearchResult<i32>;

        // Win > eval > Draw > Loss
        assert!(R::Terminal(Win) > R::NonTerminal(1));
        assert!(R::NonTerminal(1) > R::NonTerminal(-1));
        assert!(R::NonTerminal(-1) > R::Terminal(Draw));
        assert!(R::Terminal(Draw) > R::Terminal(Loss));
    }
}
