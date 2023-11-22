use super::{encoding::TensorEncodeable, service::DeepCmpService};
use crate::core::{position::Position, value::Value};
use core::hash::Hash;
use std::{cmp::Ordering, sync::Arc};

#[derive(Clone)]
pub struct DeepCmpValue<Position> {
    pub service: Arc<DeepCmpService<Position>>,
    pub position: Position,
    pub flip_point_of_view: bool,
}

impl<P> Value for DeepCmpValue<P>
where
    P: Position + TensorEncodeable + Hash + Clone + PartialEq + Eq + Hash + Sync + Send,
{
    fn compare(&self, other: &DeepCmpValue<P>) -> Ordering {
        // We should only be comparing values from the same model
        assert!(Arc::ptr_eq(&self.service, &other.service));

        // This is a theory, not sure if it's true
        assert_eq!(self.flip_point_of_view, other.flip_point_of_view);

        let mut cmp = self.service.compare(&self.position, &other.position);

        if self.flip_point_of_view {
            cmp = cmp.reverse();
        }

        cmp
    }

    fn reverse(&self) -> DeepCmpValue<P> {
        DeepCmpValue {
            service: self.service.clone(),
            position: self.position.clone(),
            flip_point_of_view: !self.flip_point_of_view,
        }
    }
}
