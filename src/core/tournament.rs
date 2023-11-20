use std::{
    fmt::Debug,
    sync::{Arc, Mutex},
};

use crate::core::{
    agent::Agent,
    position::Position,
    r#match::{play_match, MatchOutcome},
};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

#[derive(Debug, Clone)]
struct Entry {
    total: usize,
    wins_1: usize,
    wins_2: usize,
    draws: usize,
}

impl Entry {
    fn new() -> Self {
        Self {
            total: 0,
            wins_1: 0,
            wins_2: 0,
            draws: 0,
        }
    }

    fn add(&mut self, result: MatchOutcome) {
        self.total += 1;

        match result {
            MatchOutcome::WinAgent1 => self.wins_1 += 1,
            MatchOutcome::WinAgent2 => self.wins_2 += 1,
            MatchOutcome::Draw => self.draws += 1,
        }
    }
}

pub struct TournamentResult {
    results: Vec<Vec<Entry>>,
}

impl TournamentResult {}

pub fn tournament<P, A>(agents: Vec<A>, matches_per_pair: usize)
where
    P: Position,
    A: Fn() -> Box<dyn Agent<P>> + Sync + Send,
{
    let results = vec![vec![Entry::new(); agents.len()]; agents.len()];
    let results = Arc::new(Mutex::new(results));

    itertools::iproduct!(0..agents.len(), 0..agents.len())
        .collect::<Vec<(usize, usize)>>()
        .repeat(matches_per_pair)
        .into_par_iter()
        .for_each(|(i, j)| {
            let mut agent1 = agents[i]();
            let mut agent2 = agents[j]();

            let result = play_match(agent1.as_mut(), agent2.as_mut(), None);

            results.lock().unwrap()[i][j].add(result);
        });

    println!("{:?}", results);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{core::agent::RandomAgent, games::mnk::TicTacToe};

    #[test]
    fn test_tournament() {
        let agents = vec![
            || Box::new(RandomAgent {}) as Box<dyn Agent<TicTacToe>>,
            || Box::new(RandomAgent {}) as Box<dyn Agent<TicTacToe>>,
        ];
        tournament(agents, 100000);
    }
}
