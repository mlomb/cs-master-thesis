use super::{encoding::TensorEncodeable, service::DeepCmpService, value::DeepCmpValue};
use crate::core::{evaluator::PositionEvaluator, position::Position};
use std::{cell::RefCell, rc::Rc};

pub struct DeepCmpEvaluator<Position> {
    service: Rc<RefCell<DeepCmpService<Position>>>,
}

impl<Position> DeepCmpEvaluator<Position> {
    pub fn new(service: Rc<RefCell<DeepCmpService<Position>>>) -> Self {
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
            point_of_view: false,
        }
    }
}
