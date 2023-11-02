use std::ops::Neg;

/// Outcome of a match, from the POV of the player that should have played next
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Outcome {
    Loss,
    Draw,
    Win,
}

/// Negation of the status. Swaps (Win <--> Loss)
impl Neg for Outcome {
    type Output = Outcome;

    fn neg(self) -> Outcome {
        use Outcome::*;

        match self {
            Loss => Win,
            Draw => Draw,
            Win => Loss,
        }
    }
}
