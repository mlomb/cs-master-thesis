use crate::core::{
    agent::Agent,
    position::Position,
    r#match::{play_match, MatchOutcome},
};
use indicatif::{ProgressBar, ProgressStyle};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::fmt;
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
            .filter(|(i, j)| i != j)
            .collect::<Vec<(usize, usize)>>()
            .repeat(self.num_matches);

        if let Some(pb) = &progress_bar {
            pb.set_length(pairs.len() as u64);
        }

        if self.parallel {
            pairs.into_par_iter().for_each(run_match);
        } else {
            pairs.into_iter().for_each(run_match);
        }

        let results = results.lock().unwrap().clone();

        if let Some(pb) = &progress_bar {
            pb.finish();
        }

        TournamentResult {
            results,
            names: self.names.clone(),
            num_matches: self.num_matches,
        }
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
    names: Vec<String>,
    num_matches: usize,
}

impl TournamentResult {
    pub fn total_counts(&self, agent_name: &str) -> (usize, usize, usize, usize) {
        let mut wins = 0;
        let mut losses = 0;
        let mut draws = 0;

        let agent_index = self
            .names
            .iter()
            .position(|name| name == agent_name)
            .unwrap();

        // all values in the row are where agent plays first
        for entry in &self.results[agent_index] {
            wins += entry.wins_1;
            losses += entry.wins_2;
            draws += entry.draws;
        }
        // and the columns where agent plays second
        for row in &self.results {
            let entry = &row[agent_index];
            wins += entry.wins_2;
            losses += entry.wins_1;
            draws += entry.draws;
        }

        let total = wins + losses + draws;

        (wins, losses, draws, total)
    }

    pub fn win_rate(&self, agent_name: &str) -> f64 {
        let (wins, _, draws, total) = self.total_counts(agent_name);
        (wins as f64 + draws as f64 / 2.0) / total as f64
    }
}

impl fmt::Display for TournamentResult {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use tabled::{builder::*, settings::object::Segment, settings::*};

        let mut names_padded = self.names.clone();
        names_padded.insert(0, "".to_string());
        names_padded.insert(0, format!("N={}", self.num_matches));

        let mut builder = Builder::new();
        builder.push_record(["Win/Loss/Draw (WR%)", "", "Plays 2nd"]);
        builder.push_record(names_padded);

        for (i, row) in self.results.iter().enumerate() {
            let mut record = vec!["Plays 1st".to_string(), self.names[i].clone()];

            for (j, entry) in row.iter().enumerate() {
                if i == j {
                    let (wins, losses, draws, total) = self.total_counts(&self.names[i]);
                    record.push(format!(
                        "{}/{}/{} ({:.1}%)",
                        wins,
                        losses,
                        draws,
                        (wins as f64 + draws as f64 / 2.0) / total as f64 * 100.0
                    ));
                } else {
                    record.push(format!(
                        "{}/{}/{} ({:.1}%)",
                        entry.wins_1,
                        entry.wins_2,
                        entry.draws,
                        ((entry.wins_1 as f64 + entry.draws as f64 / 2.0)
                            / (entry.wins_1 + entry.wins_2 + entry.draws) as f64)
                            * 100.0
                    ));
                }
            }

            builder.push_record(record);
        }

        let mut table = builder.build();
        table
            .with(Style::ascii())
            .with(Color::FG_BRIGHT_WHITE)
            .with(
                Modify::new((0, 0)) // W/L/D
                    .with(Span::column(2))
                    .with(Alignment::center())
                    .with(Alignment::center_vertical()),
            )
            .with(
                Modify::new((1, 0)) // N=
                    .with(Span::column(2))
                    .with(Alignment::center())
                    .with(Alignment::center_vertical()),
            )
            // Plays second
            .with(
                Modify::new((0, 2))
                    .with(Span::column(999))
                    .with(Alignment::center())
                    .with(Alignment::center_vertical())
                    .with(Color::FG_BRIGHT_WHITE),
            )
            // Plays first
            .with(
                Modify::new((2, 0))
                    .with(Span::row(999))
                    .with(Alignment::center())
                    .with(Alignment::center_vertical()),
            )
            // agent names
            .with(Modify::new(Segment::new(2.., 2..)).with(Color::FG_BRIGHT_BLUE))
            .with(Modify::new(Segment::new(1..=1, 1..)).with(Color::BOLD | Color::FG_BRIGHT_YELLOW))
            .with(
                Modify::new(Segment::new(2.., 1..=1)).with(Color::BOLD | Color::FG_BRIGHT_YELLOW),
            );
        for i in 0..self.names.len() {
            table.with(
                Modify::new(Segment::new(i + 2..=i + 2, i + 2..=i + 2))
                    .with(Color::BOLD | Color::FG_CYAN),
            );
        }

        write!(f, "{}", table.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{core::agent::RandomAgent, games::mnk::TicTacToe};

    #[test]
    fn test_tournament() {
        let res1 = Tournament::<TicTacToe>::new()
            .add_agent("agent1", &|| Box::new(RandomAgent {}))
            .add_agent("agent2", &|| Box::new(RandomAgent {}))
            .num_matches(10000)
            .show_progress(true)
            .use_parallel(true)
            .run();

        println!("{:}", res1);
    }
}
