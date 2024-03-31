use super::{model::NnueModel, tensor::Tensor};
use shakmaty::{Board, Color, Move};
use std::{cell::RefCell, rc::Rc};

pub struct NnueAccumulator {
    accumulation: [Tensor<i16>; 2], // indexed by perspective (Color as usize)

    nnue_model: Rc<RefCell<NnueModel>>,

    // buffers for storing temporary feature indexes
    // find an alternative? since PositionStack is creating lots of these
    buffer1: Vec<u16>,
    buffer2: Vec<u16>,
}

impl NnueAccumulator {
    pub fn new(nnue_model: Rc<RefCell<NnueModel>>) -> Self {
        let num_ft = nnue_model.borrow().get_num_ft();
        NnueAccumulator {
            nnue_model,
            accumulation: [Tensor::zeros(num_ft), Tensor::zeros(num_ft)],
            buffer1: Vec::with_capacity(1000),
            buffer2: Vec::with_capacity(1000),
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

        self.buffer1.clear();

        // gather active features
        feature_set.active_features(board, perspective, &mut self.buffer1);

        // refresh the accumulator
        nnue_model.refresh_accumulator(&self.accumulation[perspective as usize], &self.buffer1);
    }

    pub fn update(&mut self, board: &Board, mov: &Move, perspective: Color) {
        if self
            .nnue_model
            .borrow()
            .get_feature_set()
            .requires_refresh(board, &mov, perspective)
        {
            // make full refresh :(
            self.refresh(board, perspective);
        } else {
            let nnue_model = self.nnue_model.borrow();
            let feature_set = nnue_model.get_feature_set();

            // update the accumulator
            let mut added_features = &mut self.buffer1;
            let mut removed_features = &mut self.buffer2;

            added_features.clear();
            removed_features.clear();

            // gather changed features
            feature_set.changed_features(
                board,
                &mov,
                perspective,
                &mut added_features,
                &mut removed_features,
            );

            // do the math
            nnue_model.update_accumulator(
                &self.accumulation[perspective as usize],
                &added_features,
                &removed_features,
            );
        }
    }

    pub fn copy_from(&mut self, other: &NnueAccumulator) {
        self.accumulation[0]
            .as_mut_slice()
            .copy_from_slice(other.accumulation[0].as_slice());
        self.accumulation[1]
            .as_mut_slice()
            .copy_from_slice(other.accumulation[1].as_slice());
    }
}
