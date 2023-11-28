use std::sync::Arc;

use ort::CPUExecutionProvider;
use thesis::{
    core::{
        agent::{Agent, RandomAgent},
        position::Position,
        r#match::play_match,
        tournament::Tournament,
    },
    games::{connect4::Connect4, connect4_strat::Connect4BasicAgent},
    nn::deep_cmp::{agent::DeepCmpAgent, service::DeepCmpService, DeepCmpTrainer},
};

fn main() -> ort::Result<()> {
    println!("Hello, world!");

    ort::init()
        .with_execution_providers([CPUExecutionProvider::default().build()])
        .with_name("deep_cmp")
        .commit()?;

    if false {
        let mut trainer = DeepCmpTrainer::<Connect4>::new(5000, 1024);

        for _i in 0..100000 {
            trainer.generate_samples();
            trainer.train();
            trainer.evaluate();
            std::thread::sleep(std::time::Duration::from_secs(1));
        }
    } else if true {
        let initial_model = Arc::new(DeepCmpService::new("models/initial/onnx_model.onnx"));
        let depth_model = Arc::new(DeepCmpService::new("models/depth/onnx_model.onnx"));
        //let best_fc_model = Arc::new(DeepCmpService::new("models/best_fc/onnx_model.onnx"));
        let best_uq_model = Arc::new(DeepCmpService::<Connect4>::new(
            "models/best_uq/onnx_model.onnx",
        ));

        let closure_initial =
            move || Box::new(DeepCmpAgent::new(initial_model.clone(), [8, 8])) as Box<dyn Agent<_>>;
        let closure_depth =
            move || Box::new(DeepCmpAgent::new(depth_model.clone(), [8, 8])) as Box<dyn Agent<_>>;
        /*
        let closure_best_fc = move || {
            Box::new(DeepCmpAgent::new(best_fc_model.clone(), [8, 8])) as Box<dyn Agent<_>>
        };
        */
        let closure_best_uq =
            move || Box::new(DeepCmpAgent::new(best_uq_model.clone(), [8, 8])) as Box<dyn Agent<_>>;

        let res = Tournament::<Connect4>::new()
            //.add_agent("random", &|| Box::new(RandomAgent {}))
            .add_agent("alphabeta", &|| Box::new(Connect4BasicAgent {}))
            .add_agent("initial", &closure_initial)
            .add_agent("depth", &closure_depth)
            //.add_agent("best_fc", &closure_best_fc)
            //.add_agent("best_uq", &closure_best_uq)
            .num_matches(40)
            .show_progress(true)
            .use_parallel(true)
            .run();

        println!("{}", res);
    } else if false {
        let initial_model = Arc::new(DeepCmpService::<Connect4>::new(
            "models/depth/onnx_model.onnx",
        ));
        let depth_model = Arc::new(DeepCmpService::new("models/best_uq/onnx_model.onnx"));
        let mut agent1 = DeepCmpAgent::new(depth_model.clone(), []);
        let mut agent2 = DeepCmpAgent::new(initial_model.clone(), []);
        let mut agent3 = Connect4BasicAgent {};

        let r = agent3.next_action(&Connect4::initial());
        dbg!(r);

        let mut history = Vec::new();

        play_match::<Connect4>(&mut agent3, &mut agent1, Some(&mut history));

        let mut who_plays = "X";
        for pos in history.iter() {
            println!("plays {}", who_plays);
            println!("{}", pos);
            who_plays = if who_plays == "X" { "O" } else { "X" };
        }
    } else {
        let model = Arc::new(DeepCmpService::<Connect4>::new(
            "models/chk_100/onnx_model.onnx",
        ));

        use tabled::{builder::*, settings::object::Segment, settings::*};

        let mut builder = Builder::new();
        builder.push_record(["", "0", "1", "2", "3", "4", "5", "6"]);

        for i in 0..7 {
            let mut row = Vec::new();
            row.push(i.to_string());

            for j in 0..7 {
                if i != j {
                    let cmp = model
                        .compare(
                            &Connect4::initial().apply_action(&i),
                            &Connect4::initial().apply_action(&j),
                        )
                        .reverse();

                    row.push(format!("{:?}", cmp));
                } else {
                    row.push("-".to_string());
                }
            }
            builder.push_record(row);
        }

        println!("{}", builder.build());
    }

    // model_management::Pepe::new("models").latest();

    Ok(())
}
