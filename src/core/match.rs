use super::{agent::Agent, outcome::Outcome, position::Position};

/// The outcome of a two-agent match
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
pub enum MatchOutcome {
    WinAgent1,
    WinAgent2,
    Draw,
    // Timeout,
}

/// Play a match between two agents.
/// Agent 1 plays first.
pub fn play_match<'a, P>(
    agent1: &'a mut dyn Agent<P>,
    agent2: &'a mut dyn Agent<P>,
    mut history: Option<&mut Vec<P>>,
) -> MatchOutcome
where
    P: Position,
{
    let mut position = P::initial();
    let mut who_plays = true;

    if let Some(ref mut vec) = history {
        // store the initial board
        vec.push(position.clone());
    }

    while let None = position.status() {
        let chosen_action = if who_plays {
            &mut *agent1
        } else {
            &mut *agent2
        }
        .next_action(&position)
        .expect("agent to return action");

        position = position.apply_action(&chosen_action);
        who_plays = !who_plays;

        if let Some(ref mut vec) = history {
            vec.push(position.clone());
        }
    }

    match position.status().unwrap() {
        // We expect a loss, since the POV is changed after the last move
        // WLWLWLWL
        //        ↑
        // LWLWLWL
        //       ↑
        Outcome::Win => panic!("we assume that the last player who played, wins"),
        Outcome::Draw => MatchOutcome::Draw,
        Outcome::Loss => {
            if who_plays {
                MatchOutcome::WinAgent2
            } else {
                MatchOutcome::WinAgent1
            }
        }
    }
}
