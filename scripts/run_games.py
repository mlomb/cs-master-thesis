import sys
sys.path.append("lib")

import os
import argparse
from glob import glob

from lib.paths import ENGINE_BIN, STOCKFISH_BIN
from lib.games import Engine, run_games

def main():
    parser = argparse.ArgumentParser(description="Run games and determine the Elo of a bunch of networks")

    parser.add_argument("--networks", default="./checkpoints", type=str)
    parser.add_argument("--n", default=1000, type=int)

    parser.add_argument("--pgn_file", default="./games.pgn", type=str)
    parser.add_argument("--concurrency", default=16, type=int)

    config = parser.parse_args()

    # read networks
    networks = glob(f"{config.networks}/*.nn")

    if len(networks) == 0:
        print("No networks found in", config.networks)
        return
    print(f"Found {len(networks)} networks")

    engines = [
        Engine(
            name=os.path.basename(nn),
            cmd=ENGINE_BIN,
            args=[f"--nn={os.path.abspath(nn)}"]
        )
        for nn in networks
    ]

    # engines.append(Engine(name="stockfish-elo2650", cmd=STOCKFISH_BIN, elo=2650))

    run_games(
        engines,
        n=config.n,
        concurrency=config.concurrency,
        pgn_file=config.pgn_file
    )

if __name__ == '__main__':
    main()
