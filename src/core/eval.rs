use super::position::{Position, Status};

pub trait Eval<A, P: Position<A>, Value> {
    fn eval(state: P) -> Value;
    fn terminal_value(&self) -> Value;
}
