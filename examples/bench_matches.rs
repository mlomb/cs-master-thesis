use indicatif::ProgressBar;
use rayon::prelude::*;
use thesis::{
    core::{agent::RandomAgent, r#match::play_match},
    games::connect4::Connect4,
};

fn main() {
    let n = 1 * 1000000;

    {
        let start = std::time::Instant::now();
        let pb = ProgressBar::new(n);

        for _ in 0..n {
            let mut agent1 = RandomAgent {};
            let mut agent2 = RandomAgent {};
            play_match::<Connect4>(&mut agent1, &mut agent2, None);
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
            play_match::<Connect4>(&mut agent1, &mut agent2, None);
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
                        play_match::<Connect4>(&mut agent1, &mut agent2, None);
                        pb.inc(1);
                    }
                });
            }
        });

        pb.finish();
        println!("elapsed: {:?}", start.elapsed());
    }
}
