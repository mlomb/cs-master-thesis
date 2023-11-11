use crate::core::evaluator::PositionEvaluator;

use super::value::{DeepCmpValue, Pair};
use ort::Session;
use std::{cmp::Ordering, collections::HashMap};

pub struct DeepCmpEvaluator<Position> {
    session: Session,
    inferences: usize,
    hs: HashMap<Pair<Position>, Ordering>,

    hits: usize,
    misses: usize,
}

impl<Position> DeepCmpEvaluator<Position> {
    pub fn new(session: Session) -> Self {
        Self {
            session: session,
            inferences: 0,
            hs: HashMap::new(),
            hits: 0,
            misses: 0,
        }
    }
}

impl<Position> PositionEvaluator<Position, DeepCmpValue<Position>> for DeepCmpEvaluator<Position>
where
    Position: Clone,
{
    fn eval(&self, position: &Position) -> DeepCmpValue<Position> {
        DeepCmpValue {
            evaluator: &*self,
            position: position.clone(),
            point_of_view: false,
        }
    }
}
