use super::service::DeepCmpService;
use crate::{
    core::{position::Position, value::Value},
    nn::nn_encoding::TensorEncodeable,
};
use core::hash::Hash;
use std::{cell::RefCell, cmp::Ordering, rc::Rc};

#[derive(Clone)]
pub struct DeepCmpValue<Position> {
    pub service: Rc<RefCell<DeepCmpService<Position>>>,
    pub position: Position,
    pub point_of_view: bool,
}

impl<P> Value for DeepCmpValue<P>
where
    P: Position + TensorEncodeable + Hash + Clone + PartialEq + Eq + Hash,
{
    fn compare(&self, other: &DeepCmpValue<P>) -> Ordering {
        // We should only be comparing values from the same model
        assert_eq!(self.service.as_ptr(), other.service.as_ptr());

        // This is a theory, not sure if it's true
        assert_eq!(self.point_of_view, other.point_of_view);

        self.service
            .borrow_mut()
            .compare(&self.position, &other.position)
    }

    fn reverse(&self) -> DeepCmpValue<P> {
        DeepCmpValue {
            service: self.service.clone(),
            position: self.position.clone(),
            point_of_view: !self.point_of_view,
        }
    }
}
