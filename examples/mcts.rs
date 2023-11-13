use shakmaty::Chess;
use thesis::core::position::Position;
use thesis::search::mcts::mcts::{MCTSSpec, MCTS};
use thesis::search::mcts::strategy::DefaultStrategy;

struct SampleSpec;

impl MCTSSpec for SampleSpec {
    type Position = Chess;
    type Strategy = DefaultStrategy;
}

type R = MCTS<SampleSpec>;

fn main() {
    use std::time::Instant;
    let now = Instant::now();

    //let mut position = TicTacToe::from_str("XX..O..O.", 'O');
    let mut position = Chess::initial();
    let mut mcts = R::new(&position, &DefaultStrategy);

    while let None = position.status() {
        for _ in 1..1000 {
            mcts.run_iteration();
        }

        //let distr = mcts.get_action_distribution();
        let best_action = &mcts.get_best_action().unwrap().clone();

        mcts.move_root(&best_action);
        position = position.apply_action(&best_action);

        //use shakmaty::Position;
        //println!("board: \n{:?}", position.board());
        //println!("best action: {:?}", best_action);
        // println!("distribution: {:?}", distr);
    }

    println!("Result: {:?}", position.status());

    let elapsed = now.elapsed();
    println!("Elapsed: {:.2?}", elapsed);
    println!("Nodes: {:?}", unsafe {
        thesis::games::chess::NODES_CREATED
    });

    // nodes per second
    let nps = unsafe { thesis::games::chess::NODES_CREATED } as f64 / elapsed.as_secs_f64();
    println!("nodes/sec: {:.2?}", nps);
}
