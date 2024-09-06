use std::time::Duration;

use shakmaty::Color;
use vampirc_uci::{UciSearchControl, UciTimeControl};

/// Search termination conditions
pub struct SearchLimits {
    /// Depth limit, do not exceed this depth
    pub depth: Option<i32>,
    /// Nodes limit, do not visit more nodes than this
    pub nodes: Option<usize>,
    /// Time limit, do not search for longer than this
    pub time: Option<Duration>,
}

impl SearchLimits {
    /// Empty limits, no restrictions
    pub fn none() -> Self {
        SearchLimits {
            depth: None,
            nodes: None,
            time: None,
        }
    }

    /// Builds the Limit structure from the UCI library
    pub fn from_uci(
        time_control: Option<UciTimeControl>,
        search_control: Option<UciSearchControl>,
        turn: Color,
    ) -> Self {
        let time_limit = match time_control {
            None => None, // infinite
            Some(UciTimeControl::Infinite) => None,
            Some(UciTimeControl::Ponder) => unimplemented!(""),
            Some(UciTimeControl::MoveTime(fixed_time)) => fixed_time.to_std().ok(), // movetime X (ms)
            Some(UciTimeControl::TimeLeft {
                white_time,
                black_time,
                white_increment,
                black_increment,
                moves_to_go: _, // TODO: use this
            }) => {
                let white_time = white_time.map(|x| x.num_milliseconds()).unwrap_or(0);
                let black_time = black_time.map(|x| x.num_milliseconds()).unwrap_or(0);
                let white_incr = white_increment.map(|x| x.num_milliseconds()).unwrap_or(0);
                let black_incr = black_increment.map(|x| x.num_milliseconds()).unwrap_or(0);

                let (my_time, my_incr) = if turn == Color::White {
                    (white_time, white_incr)
                } else {
                    (black_time, black_incr)
                };

                // basic time management:
                // move time = increment + time left * 0.02
                let ms = my_incr as f32 + 0.02 * my_time as f32;
                Some(Duration::from_millis(ms as u64))
            }
        };

        let mut max_depth = None;
        let mut max_nodes = None;

        if let Some(ref search_control) = search_control {
            if let Some(opt_depth) = search_control.depth {
                assert!(opt_depth >= 1);
                max_depth = Some(opt_depth as i32);
            }

            if let Some(opt_nodes) = search_control.nodes {
                assert!(opt_nodes >= 1);
                max_nodes = Some(opt_nodes as usize);
            }
        }

        SearchLimits {
            depth: max_depth,
            nodes: max_nodes,
            time: time_limit.map(|t| {
                // wiggle room to not time out
                (t - Duration::from_millis(2)).max(Duration::from_millis(2))
            }),
        }
    }
}
