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

/// Accumulator for the first layer of the neural network. It tracks the features of both perspectives
pub struct NnueAccumulator {
    // indexed by perspective (Color as usize)
    accumulation: [Tensor<i16>; 2],
    features: [Vec<i8>; 2],

    nnue_model: Rc<RefCell<NnueModel>>,
}

impl NnueAccumulator {
    /// Creates an accumulator for the given NNUE model
    pub fn new(nnue_model: Rc<RefCell<NnueModel>>) -> Self {
        let num_l1 = nnue_model.borrow().get_num_features();
        let num_features = nnue_model.borrow().get_feature_set().num_features() as usize;

        NnueAccumulator {
            nnue_model,
            accumulation: [Tensor::zeros(num_l1), Tensor::zeros(num_l1)],
            features: [vec![0; num_features], vec![0; num_features]],
        }
    }

    /// Returns the forward pass result for the given perspective, based on the current accumulator state
    pub fn forward(&self, perspective: Color) -> i32 {
        self.nnue_model.borrow().forward(
            &self.accumulation[perspective as usize],
            &self.accumulation[perspective.other() as usize],
        )
    }

    /// Throw away the current accumulator state for the given perspective and refresh it based on the given position
    pub fn refresh(&mut self, pos: &Chess, perspective: Color) {
        let nnue_model = self.nnue_model.borrow();
        let feature_set = nnue_model.get_feature_set();

        let mut features = INDEX_BUFFER1.take();

        // gather active features
        features.clear();
        feature_set.active_features(pos.board(), pos.turn(), perspective, &mut features);

        // update the feature counts
        let counts = &mut self.features[perspective as usize];
        counts.iter_mut().for_each(|c| *c = 0);
        for &f in features.iter() {
            counts[f as usize] += 1;
        }

        // refresh the accumulator
        features.sort_unstable();
        features.dedup(); // do not add rows twice!
        nnue_model.refresh_accumulator(&self.accumulation[perspective as usize], &features);
    }

    /// Update the accumulator state based on the given move, for the given position and perspective
    pub fn update(&mut self, pos: &Chess, mov: &Move, perspective: Color) {
        let board = pos.board();

        if self.nnue_model.borrow().get_feature_set().requires_refresh(
            board,
            mov,
            pos.turn(),
            perspective,
        ) {
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
            pos.turn(),
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

    /// Copies the state of the given accumulator into this one
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

#[cfg(test)]
mod tests {
    use super::*;
    use shakmaty::uci::UciMove;

    #[test]
    fn test_update_refresh() {
        let model =
            NnueModel::from_memory(&include_bytes!("../../../models/best.nn").to_vec()).unwrap();
        let mut acc = NnueAccumulator::new(Rc::new(RefCell::new(model)));

        let mut pos = Chess::default();
        let line = vec![
            "e2e4", "c7c5", "c2c3", "d7d5", "d2d4", "c5d4", "c3d4", "g8f6", "b1c3",
        ];

        acc.refresh(&pos, Color::White);
        acc.refresh(&pos, Color::Black);

        for mov in line.iter() {
            let mov = UciMove::from_ascii(mov.as_bytes())
                .unwrap()
                .to_move(&pos)
                .unwrap();

            let next_pos = pos.clone().play(&mov).unwrap();

            for &persp in &[Color::White, Color::Black] {
                // apply move to the accumulator
                acc.update(&pos, &mov, persp);

                // check score after update
                let update_score = acc.forward(persp);

                // refresh the accumulator with the moved position
                acc.refresh(&next_pos, persp);

                // check score after refresh
                let refresh_score = acc.forward(persp);

                assert_eq!(update_score, refresh_score);
            }

            pos = next_pos;
        }
    }
}
