use std::{cmp::Ordering, collections::HashMap};

use ndarray::Array5;
use ort::Session;
use thesis::{core::value, games::connect4::Connect4};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NNValue {
    pos: Connect4,
    pov: bool, // should flip POV?
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

        let input = Array5::<f32>::zeros((1, 7, 6, 2, 2));
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

    fn opposite(&mut self, value: &NNValue) -> NNValue {
        NNValue {
            pos: value.pos.clone(),
            pov: !value.pov,
        }
    }
}
