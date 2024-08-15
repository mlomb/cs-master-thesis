use super::{model::NnueModel, tensor::Tensor};
use shakmaty::{Board, Color};
use std::{cell::RefCell, rc::Rc};

thread_local! {
    // buffers for storing temporary feature indexes
    static INDEX_BUFFER1: RefCell<Vec<u16>> = RefCell::new(Vec::with_capacity(1000));
    static INDEX_BUFFER2: RefCell<Vec<u16>> = RefCell::new(Vec::with_capacity(1000));
}

pub struct NnueAccumulator {
    accumulation: [Tensor<i16>; 2], // indexed by perspective (Color as usize)
    features: [Vec<u16>; 2],        // current active features

    nnue_model: Rc<RefCell<NnueModel>>,
}

impl NnueAccumulator {
    pub fn new(nnue_model: Rc<RefCell<NnueModel>>) -> Self {
        let num_ft = nnue_model.borrow().get_num_ft();
        NnueAccumulator {
            nnue_model,
            accumulation: [Tensor::zeros(num_ft), Tensor::zeros(num_ft)],
            features: [Vec::with_capacity(1000), Vec::with_capacity(1000)],
        }
    }

    pub fn forward(&self, perspective: Color) -> i32 {
        self.nnue_model.borrow().forward(
            &self.accumulation[perspective as usize],
            &self.accumulation[perspective.other() as usize],
        )
    }

    pub fn refresh(&mut self, board: &Board, perspective: Color) {
        let nnue_model = self.nnue_model.borrow();
        let feature_set = nnue_model.get_feature_set();

        let mut buffer = &mut self.features[perspective as usize];

        buffer.clear();

        // gather active features
        feature_set.active_features(board, perspective, &mut buffer);

        // refresh the accumulator
        nnue_model.refresh_accumulator(&self.accumulation[perspective as usize], &buffer);
    }

    pub fn update(&mut self, board: &Board, perspective: Color) {
        let nnue_model = self.nnue_model.borrow();
        let feature_set = nnue_model.get_feature_set();

        let prev_features = self.features[perspective as usize].clone();
        let mut next_features = &mut self.features[perspective as usize];

        next_features.clear();
        feature_set.active_features(board, perspective, &mut next_features);

        // compute diff
        let mut added_features = INDEX_BUFFER1.take();
        let mut removed_features = INDEX_BUFFER2.take();

        added_features.clear();
        removed_features.clear();

        // make diff
        // TODO: make efficient
        for &f in prev_features.iter() {
            if !next_features.contains(&f) {
                removed_features.push(f);
            }
        }
        for &f in next_features.iter() {
            if !prev_features.contains(&f) {
                added_features.push(f);
            }
        }

        // do the math
        nnue_model.update_accumulator(
            &self.accumulation[perspective as usize],
            &added_features,
            &removed_features,
        );
    }

    pub fn copy_from(&mut self, other: &NnueAccumulator) {
        self.accumulation[0]
            .as_mut_slice()
            .copy_from_slice(other.accumulation[0].as_slice());
        self.accumulation[1]
            .as_mut_slice()
            .copy_from_slice(other.accumulation[1].as_slice());
        self.features[0].clone_from(&other.features[0]);
        self.features[1].clone_from(&other.features[1]);
    }
}
