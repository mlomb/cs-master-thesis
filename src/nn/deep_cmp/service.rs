use crate::{core::position::Position, nn::nn_encoding::TensorEncodeable};
use ndarray::Axis;
use ort::Session;
use std::hash::Hash;
use std::{cmp::Ordering, collections::HashMap};

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

        // insert one axis: batch size
        let b1 = left.encode().insert_axis(Axis(0));
        let b2 = right.encode().insert_axis(Axis(0));

        let mut input = b1;
        input
            .append(Axis(3), b2.view().into_shape(b2.shape()).unwrap())
            .unwrap();

        let outputs = self.session.run(ort::inputs![input].unwrap()).unwrap();
        let data = outputs[0]
            .extract_tensor::<f32>()
            .unwrap()
            .view()
            .t()
            .into_owned();

        self.inferences += 1;

        let res = if data[[0, 0]] < data[[1, 0]] {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Less
        };

        self.hs.insert(pair, res);
        res
    }
}
