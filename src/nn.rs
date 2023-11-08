use std::{cmp::Ordering, collections::HashMap};

use ndarray::{Array4, Array5, ArrayD, Axis, IxDyn};
use ort::Session;
use thesis::{
    core::{position::Position, value},
    games::connect4::{Connect4, COLS, ROWS},
};

use crate::nn_encoding::TensorEncodeable;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NNValue {
    pos: Connect4,
    pov: bool, // should flip POV?
}

impl TensorEncodeable for Connect4 {
    fn encode(&self) -> ArrayD<f32> {
        let mut tensor = ArrayD::zeros(IxDyn(&[7, 6, 2]));
        let who_plays = self.0.who_plays();

        for row in 0..ROWS {
            for col in 0..COLS {
                if let Some(at) = self.0.get_at(row, col) {
                    tensor[[col, row, if at == who_plays { 0 } else { 1 }]] = 1.0;
                }
            }
        }

        tensor
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Pair(NNValue, NNValue);

pub struct NNEvaluator;
pub struct NNValuePolicy {
    session: Session,
    inferences: usize,
    hs: HashMap<Pair, Ordering>,

    hits: usize,
    misses: usize,
}

impl NNValuePolicy {
    pub fn new(session: Session) -> Self {
        NNValuePolicy {
            session,
            inferences: 0,
            hs: HashMap::new(),

            hits: 0,
            misses: 0,
        }
    }
}

impl thesis::core::evaluator::PositionEvaluator<Connect4, NNValue> for NNEvaluator {
    fn eval(&self, state: &Connect4) -> NNValue {
        NNValue {
            pos: state.clone(),
            pov: false,
        }
    }
}

impl value::ValuePolicy<NNValue> for NNValuePolicy {
    fn compare(&mut self, left: &NNValue, right: &NNValue) -> std::cmp::Ordering {
        let pair = Pair(left.clone(), right.clone());

        if let Some(ordering) = self.hs.get(&pair) {
            self.hits += 1;
            return *ordering;
        }

        self.misses += 1;

        // insert one axis: batch size
        let b1 = left.pos.encode().insert_axis(Axis(0));
        let b2 = right.pos.encode().insert_axis(Axis(0));

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

        println!("{}", data);

        self.inferences += 1;

        let res = if data[[0, 0]] < data[[1, 0]] {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Less
        };

        self.hs.insert(pair, res);
        res
    }

    fn opposite(&mut self, value: &NNValue) -> NNValue {
        NNValue {
            pos: value.pos.clone(),
            pov: !value.pov,
        }
    }
}
