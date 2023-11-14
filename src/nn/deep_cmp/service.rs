use crate::core::position::Position;
use ndarray::{ArrayD, Axis};
use ort::Session;
use std::{cmp::Ordering, collections::HashMap};

use super::encoding::TensorEncodeable;

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Pair<P>(P, P);

pub struct DeepCmpService<Position> {
    session: Session,
    inferences: usize,
    hs: HashMap<Pair<Position>, Ordering>,

    hits: usize,
    misses: usize,
}

impl<P> DeepCmpService<P>
where
    P: Position + TensorEncodeable + Eq + Hash,
{
    pub fn new(session: Session) -> Self {
        Self {
            session: session,
            inferences: 0,
            hs: HashMap::new(),
            hits: 0,
            misses: 0,
        }
    }

    pub fn compare(&mut self, left: &P, right: &P) -> Ordering {
        let pair = Pair(left.clone(), right.clone());

        if let Some(ordering) = self.hs.get(&pair) {
            self.hits += 1;
            return *ordering;
        }
        self.misses += 1;
        self.inferences += 1;

        let mut input_tensor = ArrayD::zeros(P::input_shape());
        P::encode_input(left, right, &mut input_tensor.view_mut());

        // add batch size axis
        input_tensor = input_tensor.insert_axis(Axis(0));

        let outputs = self
            .session
            .run(ort::inputs![input_tensor].unwrap())
            .unwrap();

        let output_tensor = outputs[0].extract_tensor::<f32>().unwrap();
        let res = P::decode_output(&output_tensor.view().index_axis(Axis(0), 0));

        self.hs.insert(pair, res);
        res
    }
}
