use std::sync::mpsc::{self, Receiver, Sender};

use indicatif::ProgressBar;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use thesis::{
    core::{agent::Agent, position::Position, r#match::play_match},
    games::connect4::Connect4,
};

fn main() {
    struct RandomAgent {}
    unsafe impl Send for RandomAgent {}
    unsafe impl Sync for RandomAgent {}
    impl Agent<Connect4> for RandomAgent {
        fn next_action(&mut self, position: &Connect4) -> Option<usize> {
            position
                .valid_actions()
                .choose(&mut rand::thread_rng())
                .cloned()
        }
    }

    let n = 2 * 1000000;

    {
        let start = std::time::Instant::now();
        let pb = ProgressBar::new(n);

        for _ in 0..n {
            let mut agent1 = RandomAgent {};
            let mut agent2 = RandomAgent {};
            play_match(&mut agent1, &mut agent2, None);
            pb.inc(1);
        }

        pb.finish();
        println!("elapsed: {:?}", start.elapsed());
    }

    {
        let start = std::time::Instant::now();
        let pb = ProgressBar::new(n);

        (0..n).into_par_iter().for_each(|_| {
            let mut agent1 = RandomAgent {};
            let mut agent2 = RandomAgent {};
            play_match(&mut agent1, &mut agent2, None);
            pb.inc(1);
        });
        pb.finish();
        println!("elapsed: {:?}", start.elapsed());
    }

    {
        let start = std::time::Instant::now();
        let pb = ProgressBar::new(n);

        std::thread::scope(|s| {
            for _ in 0..8 {
                let pb = &pb;
                s.spawn(move || {
                    for _ in 0..n / 8 {
                        let mut agent1 = RandomAgent {};
                        let mut agent2 = RandomAgent {};
                        play_match(&mut agent1, &mut agent2, None);
                        pb.inc(1);
                    }
                });
            }
        });

        pb.finish();
        println!("elapsed: {:?}", start.elapsed());
    }
}
