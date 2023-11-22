use super::{encoding::TensorEncodeable, service::DeepCmpService, value::DeepCmpValue};
use crate::core::{evaluator::PositionEvaluator, position::Position};
use std::sync::Arc;

pub struct DeepCmpEvaluator<Position> {
    service: Arc<DeepCmpService<Position>>,
}

impl<Position> DeepCmpEvaluator<Position> {
    pub fn new(service: Arc<DeepCmpService<Position>>) -> Self {
        DeepCmpEvaluator { service }
    }
}

impl<P> PositionEvaluator<P, DeepCmpValue<P>> for DeepCmpEvaluator<P>
where
    P: Position + TensorEncodeable,
{
    fn eval(&self, position: &P) -> DeepCmpValue<P> {
        DeepCmpValue {
            service: self.service.clone(),
            position: position.clone(),
            flip_point_of_view: false,
        }
    }
}
