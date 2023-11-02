/// Outcome of a match
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Outcome {
    Loss,
    Draw,
    Win,
}
