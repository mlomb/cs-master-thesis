use super::{model::NnueModel, tensor::Tensor};
use fixedbitset::FixedBitSet;
use shakmaty::{Board, Color};
use std::{cell::RefCell, rc::Rc};

thread_local! {
    // buffers for storing temporary feature indexes
    static INDEX_BUFFER1: RefCell<Vec<u16>> = RefCell::new(Vec::with_capacity(1000));
    static INDEX_BUFFER2: RefCell<Vec<u16>> = RefCell::new(Vec::with_capacity(1000));
}

pub struct NnueAccumulator {
    accumulation: [Tensor<i16>; 2], // indexed by perspective (Color as usize)
    features_bitset: [FixedBitSet; 2], // current active features

    nnue_model: Rc<RefCell<NnueModel>>,
}

impl NnueAccumulator {
    pub fn new(nnue_model: Rc<RefCell<NnueModel>>) -> Self {
        let num_l1 = nnue_model.borrow().get_num_ft();
        let num_features = nnue_model.borrow().get_feature_set().num_features();
        NnueAccumulator {
            nnue_model,
            accumulation: [Tensor::zeros(num_l1), Tensor::zeros(num_l1)],
            features_bitset: [
                FixedBitSet::with_capacity(num_features),
                FixedBitSet::with_capacity(num_features),
            ],
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

        let mut bitset = &mut self.features_bitset[perspective as usize];

        // gather active features
        bitset.clear();
        feature_set.active_features(board, perspective, &mut bitset);

        // refresh the accumulator
        nnue_model.refresh_accumulator(
            &self.accumulation[perspective as usize],
            bitset
                .ones()
                .map(|f| f as u16)
                .collect::<Vec<_>>()
                .as_slice(),
        );
    }

    pub fn update(&mut self, board: &Board, perspective: Color) {
        let nnue_model = self.nnue_model.borrow();
        let feature_set = nnue_model.get_feature_set();

        let prev_bitset = self.features_bitset[perspective as usize].clone();
        let mut next_bitset = &mut self.features_bitset[perspective as usize];

        next_bitset.clear();
        feature_set.active_features(board, perspective, &mut next_bitset);

        // compute diff
        let mut added_features = INDEX_BUFFER1.take();
        let mut removed_features = INDEX_BUFFER2.take();

        added_features.clear();
        removed_features.clear();

        // make diff
        next_bitset.difference(&prev_bitset).for_each(|f| {
            added_features.push(f as u16);
        });
        prev_bitset.difference(&next_bitset).for_each(|f| {
            removed_features.push(f as u16);
        });

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
        self.features_bitset[0].clone_from(&other.features_bitset[0]);
        self.features_bitset[1].clone_from(&other.features_bitset[1]);
    }
}
