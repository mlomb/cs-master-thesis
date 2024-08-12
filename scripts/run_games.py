import os
import argparse
from glob import glob
import random
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Run games and determine the Elo of a bunch of networks")

    parser.add_argument("--models", default="./models", type=str)
    parser.add_argument("--engine", default="../engine/target/release/engine", type=str)
    parser.add_argument("--book_file_name", default="./noob_3moves.epd", type=str)

    parser.add_argument("--pgn_file", default="./games.pgn", type=str)
    parser.add_argument("--concurrency", default=16, type=int)

    parser.add_argument("--c_chess_exe", default="./c-chess-cli", type=str)
    parser.add_argument("--ordo_exe", default="./ordo", type=str)

    config = parser.parse_args()

    # read models
    models = glob(f"{config.models}/*.nn")
    print(f"Found {len(models)} models")

    # build command
    command = []
    command += [
        config.c_chess_exe,
        '-games', '10',
        '-concurrency', f'{config.concurrency}'
    ]
    command += [
        '-openings', f'file={config.book_file_name}', 'order=random', f'srand={random.randint(0,100000000)}',
        '-repeat',
        '-pgn', f'{config.pgn_file}'
    ]
    #command += ['-engine', f'cmd={config.engine}', 'name=master1', 'nodes=20000']
    #command += ['-engine', f'cmd={config.engine}', 'name=master2', 'nodes=20000']

    for model in models:
        nn_name = os.path.basename(model).replace("(", "").replace(")", "").replace(">", "-")
        nn_path = os.path.abspath(model)
        command += ['-engine', f'cmd="{config.engine} --nn={nn_path}"', f'name={nn_name}', 'nodes=20000']

    print(" ".join(command))


if __name__ == '__main__':
    main()
