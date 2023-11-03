use super::outcome::Outcome;

/// A game position. It must contain all the information needed to continue playing. (i.e. the board, the player to move, etc.)
pub trait Position: Clone {
    /// The type of actions that can be applied to the position
    type Action: Clone + Eq;

    /// Generates an initial position for the game
    fn initial() -> Self;

    /// Lists all valid actions from the current position
    ///
    /// Note that this function must be deterministic; it must always return the same actions in the same order.
    fn valid_actions(&self) -> Vec<Self::Action>;

    /// Returns a new position after the given action is applied
    ///
    /// ⚠️ Must always change the point of view of the player even if the player won!
    /// It means that if the player that just played won, the resulting position's `status()` will report a LOSS, since the POV changed.
    fn apply_action(&self, action: &Self::Action) -> Self;

    /// Returns the status of the position from the POV of the player to move
    /// None if the game is not over
    fn status(&self) -> Option<Outcome>;
}
