use crate::core::{
    agent::Agent,
    position::Position,
    r#match::{play_match, MatchOutcome},
};
use indicatif::{ProgressBar, ProgressStyle};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::{
    fmt::Debug,
    sync::{Arc, Mutex},
};

/// A function that returns a new agent.
type AgentGenerator<P> = dyn Fn() -> Box<dyn Agent<P>> + Sync + Send;

pub struct Tournament<'a, P>
where
    P: Position,
{
    agents: Vec<&'a AgentGenerator<P>>,
    names: Vec<String>,
    num_matches: usize,
    parallel: bool,
    progress: bool,
}

impl<'a, P> Tournament<'a, P>
where
    P: Position,
{
    pub fn new() -> Self {
        Self {
            agents: Vec::new(),
            names: Vec::new(),
            num_matches: 10,
            parallel: false,
            progress: false,
        }
    }

    pub fn add_agent(mut self, name: &str, agent: &'a AgentGenerator<P>) -> Self {
        self.names.push(name.to_string());
        self.agents.push(agent);
        self
    }

    pub fn num_matches(mut self, matches_per_pair: usize) -> Self {
        self.num_matches = matches_per_pair;
        self
    }

    pub fn use_parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    pub fn show_progress(mut self, progress: bool) -> Self {
        self.progress = progress;
        self
    }

    pub fn run(&self) -> TournamentResult {
        let results = vec![vec![Entry::new(); self.agents.len()]; self.agents.len()];
        let results = Arc::new(Mutex::new(results));

        let progress_bar = if self.progress {
            let pb = ProgressBar::new(0).with_style(
                ProgressStyle::with_template(
                    "{spinner:.green} [Elapsed {elapsed_precise}] (ETA {eta}) [{bar:.cyan/blue}] {human_pos}/{human_len}  {per_sec} ",
                )
                .unwrap()
                .progress_chars("#987654321-")
            );

            Some(pb)
        } else {
            None
        };

        if let Some(pb) = &progress_bar {
            pb.set_length((self.agents.len().pow(2) * self.num_matches) as u64);
        }

        let run_match = |(i, j): (usize, usize)| {
            let mut agent1: Box<dyn Agent<P>> = self.agents[i]();
            let mut agent2: Box<dyn Agent<P>> = self.agents[j]();

            let result = play_match(agent1.as_mut(), agent2.as_mut(), None);

            results.lock().unwrap()[i][j].add(result);

            if let Some(pb) = &progress_bar {
                pb.inc(1);
            }
        };

        let pairs = itertools::iproduct!(0..self.agents.len(), 0..self.agents.len())
            .collect::<Vec<(usize, usize)>>()
            .repeat(self.num_matches);

        if self.parallel {
            pairs.into_par_iter().for_each(run_match);
        } else {
            pairs.into_iter().for_each(run_match);
        }

        let results = results.lock().unwrap().clone();

        if let Some(pb) = &progress_bar {
            pb.finish();
        }

        TournamentResult { results }
    }
}

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

#[derive(Debug)]
pub struct TournamentResult {
    results: Vec<Vec<Entry>>,
}

impl TournamentResult {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{core::agent::RandomAgent, games::mnk::TicTacToe};

    #[test]
    fn test_tournament() {
        let res1 = Tournament::<TicTacToe>::new()
            .add_agent("agent1", &|| Box::new(RandomAgent {}))
            .add_agent("agent2", &|| Box::new(RandomAgent {}))
            .num_matches(1000)
            .show_progress(true)
            .use_parallel(true)
            .run();

        println!("{:?}", res1);

        /*
        let agents = vec![
            || Box::new(RandomAgent {}) as Box<dyn Agent<TicTacToe>>,
            || Box::new(RandomAgent {}) as Box<dyn Agent<TicTacToe>>,
        ];
        let pb = ProgressBar::new(0);
        let res = tournament(agents, 100000, Some(&pb));

        println!("{:?}", res);
        */
    }
}
