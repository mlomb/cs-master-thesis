use ndarray::{ArrayViewMutD, Axis};
use rand::Rng;

use super::{encoding::TensorEncodeable, ringbuffer_set::RingBufferSet};
use crate::core::r#match::MatchOutcome;
use std::{cmp::Ordering, collections::HashSet, hash::Hash};

pub struct Samples<P> {
    win_positions: RingBufferSet<P>,
    loss_positions: RingBufferSet<P>,
    all_positions: HashSet<P>,
}

impl<P> Samples<P>
where
    P: TensorEncodeable + Eq + Hash + Clone,
{
    pub fn new(capacity: usize) -> Self {
        Self {
            win_positions: RingBufferSet::new(capacity),
            loss_positions: RingBufferSet::new(capacity),
            all_positions: HashSet::new(),
        }
    }

    pub fn add_match(&mut self, history: &Vec<P>, outcome: MatchOutcome) {
        // even positions
        for pos in history.iter().step_by(2) {
            match outcome {
                MatchOutcome::WinAgent1 => self.win_positions.insert(pos.clone()),
                MatchOutcome::WinAgent2 => self.loss_positions.insert(pos.clone()),
                _ => (),
            }
        }

        // odd positions
        for pos in history.iter().skip(1).step_by(2) {
            match outcome {
                MatchOutcome::WinAgent1 => self.loss_positions.insert(pos.clone()),
                MatchOutcome::WinAgent2 => self.win_positions.insert(pos.clone()),
                _ => (),
            }
        }

        // add all positions to the set
        self.all_positions.extend(history.iter().cloned());

        /*
        println!(
            "win: {}, loss: {} all: {}",
            self.win_positions.len(),
            self.loss_positions.len(),
            self.all_positions.len()
        );
        */
    }

    pub fn write_samples(
        &self,
        batch_size: usize,
        inputs_tensor: &mut ArrayViewMutD<f32>,
        outputs_tensor: &mut ArrayViewMutD<f32>,
    ) {
        // clear tensors
        inputs_tensor.fill(0.0);
        outputs_tensor.fill(0.0);

        let mut rng = rand::thread_rng();

        for i in 0..batch_size {
            let win_pos = self.win_positions.sample_one(&mut rng).unwrap();
            let loss_pos = self.loss_positions.sample_one(&mut rng).unwrap();

            let (left_board, ordering, right_board) = if rng.gen_bool(0.5) {
                (win_pos, Ordering::Greater, loss_pos)
            } else {
                (loss_pos, Ordering::Less, win_pos)
            };

            P::encode_input(
                left_board,
                right_board,
                &mut inputs_tensor.index_axis_mut(Axis(0), i),
            );
            P::encode_output(ordering, &mut outputs_tensor.index_axis_mut(Axis(0), i));
        }
    }
}
