use ndarray::{ArrayViewMutD, Axis};
use rand::{
    distributions::{Distribution, WeightedIndex},
    seq::SliceRandom,
    Rng,
};

use super::{encoding::TensorEncodeable, ringbuffer_set::RingBufferSet};
use crate::core::r#match::MatchOutcome;
use std::{cmp::Ordering, collections::HashSet, hash::Hash, os::windows};

pub struct SamplesDepth<P> {
    win_positions: RingBufferSet<P>,
    loss_positions: RingBufferSet<P>,
    all_positions: HashSet<P>,

    only_wins: HashSet<P>,
    only_losses: HashSet<P>,
}

impl<P> SamplesDepth<P>
where
    P: Eq + Hash + Clone,
{
    pub fn new(capacity: usize) -> Self {
        Self {
            win_positions: RingBufferSet::<P>::new(capacity),
            loss_positions: RingBufferSet::<P>::new(capacity),
            all_positions: HashSet::new(),

            only_wins: HashSet::new(),
            only_losses: HashSet::new(),
        }
    }
}

pub struct Samples<P> {
    at_depth: Vec<SamplesDepth<P>>,
}

impl<P> Samples<P>
where
    P: TensorEncodeable + Eq + Hash + Clone,
{
    pub fn new(capacity: usize) -> Self {
        let mut at_depth = Vec::new();
        // 42?
        for _ in 0..50 {
            at_depth.push(SamplesDepth::new(capacity));
        }
        Self { at_depth }
    }

    pub fn add_match(&mut self, history: &Vec<P>, outcome: MatchOutcome) {
        let mut is_win = match outcome {
            MatchOutcome::WinAgent1 => true,
            MatchOutcome::WinAgent2 => false,
            MatchOutcome::Draw => return,
        };

        for (depth, pos) in history.iter().enumerate() {
            let sd = &mut self.at_depth[depth];

            if is_win {
                sd.win_positions.insert(pos.clone());
            } else {
                sd.loss_positions.insert(pos.clone());
            }

            if is_win {
                sd.only_losses.remove(pos);
            } else {
                sd.only_wins.remove(pos);
            }

            if !sd.all_positions.contains(pos) {
                if is_win {
                    sd.only_wins.insert(pos.clone());
                } else {
                    sd.only_losses.insert(pos.clone());
                }

                sd.all_positions.insert(pos.clone());
            }

            is_win = !is_win;
        }

        for d in 0..20 {
            let sd = &mut self.at_depth[d];
            print!("{}/{} ", sd.only_wins.len(), sd.only_losses.len());
        }
        println!();
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

        let depth_weights = (0..self.at_depth.len())
            .map(|depth| {
                let sd = &self.at_depth[depth];
                2 * sd.only_losses.len() * sd.only_wins.len()
            })
            .collect::<Vec<_>>();
        let dist = WeightedIndex::new(&depth_weights).unwrap();

        for i in 0..batch_size {
            let depth = dist.sample(&mut rng);
            let sd = &self.at_depth[depth];

            let win_pos = sd
                .only_wins
                .iter()
                .cloned()
                .collect::<Vec<P>>()
                .choose(&mut rng)
                .unwrap()
                .clone();
            let loss_pos = sd
                .only_losses
                .iter()
                .cloned()
                .collect::<Vec<_>>()
                .choose(&mut rng)
                .unwrap()
                .clone();

            // let win_pos = sd.win_positions.sample_one(&mut rng).unwrap();
            //let loss_pos = sd.loss_positions.sample_one(&mut rng).unwrap();

            let (left_board, ordering, right_board) = if rng.gen_bool(0.5) {
                (win_pos, Ordering::Greater, loss_pos)
            } else {
                (loss_pos, Ordering::Less, win_pos)
            };

            P::encode_input(
                &left_board,
                &right_board,
                &mut inputs_tensor.index_axis_mut(Axis(0), i),
            );
            P::encode_output(ordering, &mut outputs_tensor.index_axis_mut(Axis(0), i));
        }
    }
}
