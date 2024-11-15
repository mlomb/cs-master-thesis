import sys
sys.path.append("..")

import torch
import numpy as np
import subprocess
import tempfile
from tqdm import tqdm

from lib.serialize import NnueWriter
from lib.model import NnueModel
from lib.paths import TOOLS_BIN, DEFAULT_DATASET

N = 1_000_000

with open(DEFAULT_DATASET) as f:
    FENS = [
        next(f).split(",")[0]
        for _ in tqdm(range(N), desc="Reading FENS")
    ]


def compute_error(fen: str, nn: str):
    output = subprocess.check_output([
        TOOLS_BIN,
        "info",
        f"--feature-set=hv",
        f"--fen={fen}",
        f"--nn={nn}",
    ])
    fs_size, features_pov, features_opp, output = output.decode("utf-8").strip().splitlines()

    fs_size = int(fs_size)
    features_pov = np.array(list(map(int, features_pov.split())))
    features_opp = np.array(list(map(int, features_opp.split())))
    output = int(output)
    
    # print("FEN:", fen)
    # print("Feature set size:", fs_size)
    # print("Features pov:", features_pov)
    # print("Features opp:", features_opp)
    # print("Eval:", eval)

    input = torch.zeros((2, 768))
    input[0, features_pov] = 1
    input[1, features_opp] = 1

    expected_output = model(input).item()

    return output, expected_output
    


model = NnueModel(768, l1_size=512)
model.load_state_dict(torch.load('../data/256-4-eval_16384_(hv[768]→512)x2→32→1.pth'))
model.clip_weights()

writer = NnueWriter(model, "hv")
with tempfile.NamedTemporaryFile() as nn:
    nn.write(writer.buf)

    with open("errors.csv", "w+") as f:
        f.write("output,expected_output\n")

        for fen in tqdm(FENS, desc="Computing errors"):
            output, expected_output = compute_error(fen, nn.name)
            f.write(f"{output},{expected_output}\n")

