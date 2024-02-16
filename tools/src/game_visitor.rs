use clap::Args;
use pgn_reader::{RawHeader, SanPlus, Skip, Visitor};
use shakmaty::{Chess, Position};

#[derive(Args)]
pub struct VisitorConfig {
    /// Only accept positions that are at least this many plies deep
    #[arg(long, value_name = "min-ply", default_value = "12")]
    min_ply: usize,

    /// Only accept games where both player have at least this elo
    #[arg(long, value_name = "min-elo")]
    min_elo: Option<usize>,
}

pub struct GameVisitor {
    // Configuration
    config: VisitorConfig,

    /// All positions for the current game
    positions: Vec<Chess>,

    // information about the current game
    event: String,
    termination: String,
    white_elo: usize,
    black_elo: usize,
}

impl GameVisitor {
    pub fn new(config: VisitorConfig) -> Self {
        GameVisitor {
            config,

            positions: vec![Chess::default()], // start pos

            event: "".to_string(),
            termination: "".to_string(),
            white_elo: 0,
            black_elo: 0,
        }
    }
}

impl Visitor for GameVisitor {
    type Result = Option<Vec<Chess>>;

    fn begin_game(&mut self) {
        self.event = "".to_string();
        self.termination = "".to_string();
        self.white_elo = 0;
        self.black_elo = 0;
        self.positions.truncate(1); // only keep starting board
    }

    fn header(&mut self, _key: &[u8], _value: RawHeader<'_>) {
        let key = String::from_utf8_lossy(_key);
        let value = String::from_utf8_lossy(_value.as_bytes());

        if key == "Event" {
            self.event = value.to_string();
        } else if key == "Termination" {
            self.termination = value.to_string();
        } else if key == "BlackElo" {
            self.black_elo = value.parse().unwrap_or(0);
        } else if key == "WhiteElo" {
            self.white_elo = value.parse().unwrap_or(0);
        }
    }

    fn end_headers(&mut self) -> Skip {
        let min_elo = self.config.min_elo.unwrap_or(0);

        let keep =
            // keep rated classical games (disabled, accept blitz too)
            //self.event.starts_with("Rated Classical") &&
            // keep normal terminations (excl. Abandoned, Time forfeit, Rules infraction)
            self.termination == "Normal" &&
            // keep games with good elo
            self.white_elo >= min_elo &&
            self.black_elo >= min_elo;

        Skip(!keep)
    }

    fn begin_variation(&mut self) -> Skip {
        Skip(true)
    }

    fn san(&mut self, _san_plus: SanPlus) {
        let pos = self.positions.last().unwrap().clone();
        let mov = _san_plus.san.to_move(&pos).unwrap();
        self.positions.push(pos.play(&mov).unwrap());
    }

    fn end_game(&mut self) -> Self::Result {
        // game too short?
        if self.positions.len() <= 20 {
            // note: skipped games go through here too
            return None;
        }

        let range = self.config.min_ply..self.positions.len() - 1; // dont pick last
        if range.is_empty() {
            // no feasible positions
            return None;
        }

        Some(self.positions[range].to_vec())
    }
}
