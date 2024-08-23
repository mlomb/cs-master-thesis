use super::{model::NnueModel, tensor::Tensor};
use shakmaty::{Chess, Color, Move, Position};
use std::{cell::RefCell, rc::Rc};

thread_local! {
    // buffers for storing temporary feature indexes
    static INDEX_BUFFER1: RefCell<Vec<u16>> = RefCell::new(Vec::with_capacity(128));
    static INDEX_BUFFER2: RefCell<Vec<u16>> = RefCell::new(Vec::with_capacity(128));
    static INDEX_BUFFER3: RefCell<Vec<u16>> = RefCell::new(Vec::with_capacity(128));
    static INDEX_BUFFER4: RefCell<Vec<u16>> = RefCell::new(Vec::with_capacity(128));
}

pub struct NnueAccumulator {
    accumulation: [Tensor<i16>; 2], // indexed by perspective (Color as usize)
    features: [Vec<i8>; 2],

    nnue_model: Rc<RefCell<NnueModel>>,
}

impl NnueAccumulator {
    pub fn new(nnue_model: Rc<RefCell<NnueModel>>) -> Self {
        let num_l1 = nnue_model.borrow().get_num_ft();
        let num_features = nnue_model.borrow().get_feature_set().num_features();
        NnueAccumulator {
            nnue_model,
            accumulation: [Tensor::zeros(num_l1), Tensor::zeros(num_l1)],
            features: [vec![0; num_features], vec![0; num_features]],
        }
    }

    pub fn forward(&self, perspective: Color) -> i32 {
        self.nnue_model.borrow().forward(
            &self.accumulation[perspective as usize],
            &self.accumulation[perspective.other() as usize],
        )
    }

    pub fn refresh(&mut self, pos: &Chess, perspective: Color) {
        let nnue_model = self.nnue_model.borrow();
        let feature_set = nnue_model.get_feature_set();

        let mut features = INDEX_BUFFER1.take();

        // gather active features
        features.clear();
        feature_set.active_features(pos.board(), perspective, &mut features);

        // update the feature counts
        let counts = &mut self.features[perspective as usize];
        counts.iter_mut().for_each(|c| *c = 0);
        for &f in features.iter() {
            counts[f as usize] += 1;
        }

        // refresh the accumulator
        features.sort_unstable();
        features.dedup();
        nnue_model.refresh_accumulator(&self.accumulation[perspective as usize], &features);
    }

    pub fn update(&mut self, pos: &Chess, mov: &Move, perspective: Color) {
        let board = pos.board();

        if self
            .nnue_model
            .borrow()
            .get_feature_set()
            .requires_refresh(board, mov, perspective)
        {
            let mut next_pos = pos.clone();
            next_pos.play_unchecked(mov);
            self.refresh(&next_pos, perspective);
            return;
        }

        let nnue_model = self.nnue_model.borrow();
        let feature_set = nnue_model.get_feature_set();
        let counts = &mut self.features[perspective as usize];

        let mut added_features = INDEX_BUFFER1.take();
        let mut removed_features = INDEX_BUFFER2.take();

        added_features.clear();
        removed_features.clear();

        feature_set.changed_features(
            board,
            mov,
            perspective,
            &mut added_features,
            &mut removed_features,
        );

        // compute which rows to add and remove
        // based on the feature counts after applying the modifications
        let mut added_rows = INDEX_BUFFER3.take();
        let mut removed_rows = INDEX_BUFFER4.take();

        added_rows.clear();
        removed_rows.clear();

        for &f in added_features.iter() {
            if counts[f as usize] == 0 {
                added_rows.push(f);
            }
            counts[f as usize] += 1;
        }
        for &f in removed_features.iter() {
            counts[f as usize] -= 1;
            if counts[f as usize] == 0 {
                removed_rows.push(f);
            }
        }

        // do the math
        nnue_model.update_accumulator(
            &self.accumulation[perspective as usize],
            &added_rows,
            &removed_rows,
        );
    }

    pub fn copy_from(&mut self, other: &NnueAccumulator) {
        self.accumulation[0]
            .as_mut_slice()
            .copy_from_slice(other.accumulation[0].as_slice());
        self.accumulation[1]
            .as_mut_slice()
            .copy_from_slice(other.accumulation[1].as_slice());
        self.features[0].copy_from_slice(&other.features[0]);
        self.features[1].copy_from_slice(&other.features[1]);
    }
}
