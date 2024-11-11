# Feature set analysis for chess NNUE networks

### **Abstract**

> Historically, chess engines have used highly complex functions to evaluate chess positions. Recently, efficiently updatable neural networks (NNUE) have displaced these functions without the need of human knowledge. The input of these networks are called feature sets and they take advantage of the order in which positions are evaluated in a depth-first search to save computation.
> 
> In this thesis, I develop a classical chess engine, where the evaluation function is replaced by a NNUE network trained with a pipeline created from scratch. The main goal of this thesis is to test novel feature sets that can improve performance. Additionally, a way of training the networks is tried using a method proposed years ago but with a higher volume and quality of data available in the post-NNUE era.

**The dissertation document can be found here: [thesis.pdf](https://mlomb.github.io/cs-master-thesis-doc/thesis.pdf).**

----

The bot can be challenged on Lichess: [https://lichess.org/@/LimboBot](https://lichess.org/@/LimboBot)!

[![lichess-bullet](https://lichess-shield.vercel.app/api?username=LimboBot&format=bullet)](https://lichess.org/@/LimboBot/perf/bullet)
[![lichess-blitz](https://lichess-shield.vercel.app/api?username=LimboBot&format=blitz)](https://lichess.org/@/LimboBot/perf/blitz)

# Structure

The code is structured in the following way:

- `engine/`: The chess engine.
- `models/`: The trained NNUE models.
- `nn_viz/`: Network visualization website ([https://mlomb.github.io/cs-master-thesis](https://mlomb.github.io/cs-master-thesis))
- `nn/`: Implementation of features sets and NNUE.
- `scripts/`: Training, testing and analysis scripts.
- `tools/`: Tools for processing samples (batching for training) and gather information and statistics.

Most of the code is written in Rust and Python.

# The engine

The engine uses the [UCI protocol](https://www.shredderchess.com/chess-features/uci-universal-chess-interface.html) to communicate via standard input. Run the crate `engine` to start. By default, it will use the best network in the `models/` directory.

```bash
$ cd engine
$ cargo run --release
info string Using embedded NNUE
info string NNUE net: (hv[768]→512)x2→32→1
info string NNUE size: 426593 params
go
info depth 1 time 5 nodes 24 evals 23 score cp 117 pv d2d4 
info depth 2 time 14 nodes 192 evals 168 score cp 63 pv d2d4 d7d5 
info depth 3 time 14 nodes 460 evals 369 score cp 76 pv d2d4 d7d5 g1f3 
info depth 4 time 15 nodes 1126 evals 928 score cp 54 pv e2e4 e7e5 b1c3 b8c6 
info depth 5 time 16 nodes 3046 evals 2518 score cp 53 pv e2e4 e7e5 b1c3 b8c6 g1f3 
info depth 6 time 18 nodes 6132 evals 4978 score cp 58 pv d2d4 d7d5 g1f3 b8c6 c1f4 g8f6 
info depth 7 time 26 nodes 20422 evals 17096 score cp 60 pv e2e4 e7e5 b1c3 b8c6 g1f3 g8f6 d2d4 
info depth 8 time 48 nodes 55899 evals 47152 score cp 60 pv e2e4 e7e5 b1c3 b8c6 g1f3 g8f6 d2d4 e5d4 
...
```

You can specify a network using `--nn`.

# Training

First, make sure to build both the `engine` and `tools` binaries:

```bash
cd engine
cargo build --release
```

and

```bash
cd tools
cargo build --release
```

Then, you can train a model by running:

```bash
cd scripts
python train.py --feature_set=all --ft_size=256 --wandb_project optional_project_name
```

There are more options available, check `train.py` for more information.