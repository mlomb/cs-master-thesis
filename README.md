# Analysis of feature sets in NNUE networks for chess engines

[poner abstract]

The bot can be challenged on Lichess: [https://lichess.org/@/LimboBot](https://lichess.org/@/LimboBot)

# Code

The code is structured in the following way:

- `engine/`: The chess engine.
- `models/`: The trained NNUE models.
- `nn_viz/`: Network visualization website.
- `nn/`: Implementation of features sets and NNUE.
- `scripts/`: Training, testing and analysis scripts.
- `tools/`: Tools for processing samples: batching for training.

Most of the code is written in Rust and Python.

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
python train.py --feature_set=hv --ft_size=256 --wandb_project optional_project_name
```
