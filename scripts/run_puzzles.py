import sys
sys.path.append("lib")

import os
import argparse
from glob import glob

from lib.paths import ENGINE_BIN, PUZZLES_DATA_1000
from lib.puzzles import Puzzles

def main():
    parser = argparse.ArgumentParser(description="Run puzzles and determine the accuracy for each network")

    parser.add_argument("--networks", default="./checkpoints", type=str)
    parser.add_argument("--puzzles", default=PUZZLES_DATA_1000, type=str)
    parser.add_argument("--concurrency", default=16, type=int)

    config = parser.parse_args()

    # read networks
    networks = glob(f"{config.networks}/*.nn")

    if len(networks) == 0:
        print("No networks found in", config.networks)
        return
    print(f"Found {len(networks)} networks")

    p = Puzzles(config.puzzles)

    for nn in networks:
        name = os.path.basename(nn)
        path = os.path.abspath(nn)

        table, moveAcc = p.measure([ENGINE_BIN, f"--nn={path}"], concurrency=config.concurrency)

        print(f"Network: {name} Accuracy: {moveAcc}")

if __name__ == '__main__':
    main()
