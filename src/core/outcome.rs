/// Outcome of a match
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
pub enum Outcome {
    Loss,
    Draw,
    Win,
}
