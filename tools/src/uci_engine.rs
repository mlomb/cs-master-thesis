use std::io::{self, BufRead, Write};
use std::process::{Child, Command, Stdio};

/// Score of a position, given by the engine
#[derive(Debug)]
pub enum Score {
    /// Centipawn
    Cp(i32),

    /// Mate/Mated in n
    Mate(i32),
}

#[derive(Debug)]
pub struct EngineResult {
    pub score: Score,
    pub best_move: String,
}

/// Extremely simple UCI engine wrapper to evaluate positions
/// https://www.wbec-ridderkerk.nl/html/UCIProtocol.html
pub struct UciEngine {
    /// Target depth for the search
    target_depth: usize,

    /// Child process
    child: Child,
}

impl UciEngine {
    pub fn new(binary: &str, target_depth: usize) -> Self {
        // TODO: isready?
        UciEngine {
            target_depth,
            child: Command::new(binary)
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .expect("Engine failed to start"),
        }
    }

    pub fn evaluate(&mut self, fen: &str) -> EngineResult {
        let stdin = self.child.stdin.as_mut().unwrap();
        let stdout = self.child.stdout.as_mut().unwrap();

        // start search
        // writeln!(stdin, "setoption name Threads value 1").unwrap();
        writeln!(stdin, "position fen {}", fen).unwrap();
        writeln!(stdin, "go depth {}", self.target_depth).unwrap();

        // now read the output
        // and wait for the depth to be reached
        let mut line = String::new();
        let mut reader = io::BufReader::new(stdout);

        let mut score = None;
        let best_move;

        loop {
            reader.read_line(&mut line).unwrap();
            let mut parts = line.split_whitespace();

            if line.starts_with("info depth") {
                parts.position(|p| p == "score").unwrap();

                let score_type = parts.nth(0).unwrap();
                let score_value = parts.nth(0).unwrap().parse::<i32>().unwrap();

                score = if score_type == "cp" {
                    Some(Score::Cp(score_value))
                } else if score_type == "mate" {
                    Some(Score::Mate(score_value))
                } else {
                    panic!("Bad score: {}", line);
                };
            } else if line.starts_with("bestmove") {
                parts.position(|p| p == "bestmove").unwrap();
                best_move = parts.nth(0).unwrap();

                // done
                break;
            }

            line.clear();
        }

        EngineResult {
            score: score.unwrap(),
            best_move: best_move.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ENGINE: &str = "./stockfish-ubuntu-x86-64-avx2";

    #[test]
    fn test_score() {
        let mut engine = UciEngine::new(ENGINE, 12);
        let res = engine.evaluate("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        println!("{:?}", res);
    }

    #[test]
    fn test_mate() {
        let mut engine = UciEngine::new(ENGINE, 12);
        let res = engine.evaluate("8/8/1q4b1/8/8/4k3/K7/8 b - - 19 68");
        println!("{:?}", res);
    }
}
