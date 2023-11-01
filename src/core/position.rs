use std::ops::Neg;

/// Status of the position from the POV of the player to move
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Outcome {
    LOSS,
    DRAW,
    WIN,
}

/// Negation of the status. Swaps WIN <--> LOSS
impl Neg for Outcome {
    type Output = Outcome;

    fn neg(self) -> Outcome {
        match self {
            Outcome::PLAYING => Outcome::PLAYING,
            Outcome::LOSS => Outcome::WIN,
            Outcome::DRAW => Outcome::DRAW,
            Outcome::WIN => Outcome::LOSS,
        }
    }
}

/// A game position. It must contain all the information needed to continue playing. (i.e. the board, the player to move, etc.)
pub trait Position<Action> {
    /// Generates an initial position for the game
    fn initial() -> Self;

    /// Lists all valid actions from the current position
    fn valid_actions(&self) -> Vec<Action>;

    /// Returns a new position after the given action is applied
    ///
    /// ⚠️ Must always change the point of view of the player even if the player won!
    /// It means that if the player that just played won, the resulting position's `status()` will report a LOSS, since the POV changed.
    fn apply_action(&self, action: Action) -> Self;

    /// Returns the status of the position from the POV of the player to move
    fn status(&self) -> Outcome;
}
