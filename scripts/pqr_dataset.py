import sys
sys.path.append("lib")

import torch
from tqdm import tqdm

from lib.batch_loader import BatchLoader
from lib.model import NnueModel, expand_batch


samples = BatchLoader(
    batch_size=16384,
    batch_threads=2,
    input="/mnt/c/datasets/raw/all.plain",
    input_offset=0,
    input_loop=True,
    feature_set="all",
    method="pqr",
    random_skipping=0.0,
)

chessmodel = NnueModel(
    num_features=768,
    l1_size=512,
    l2_size=32
)
chessmodel.load_state_dict(torch.load('./data/256-4-eval_16384_(hv[768]→512)x2→32→1.pth'))
#chessmodel.load_state_dict(torch.load('/mnt/c/Users/Lombi/Desktop/tesis/cs-master-thesis/scripts/checkpoints/1729671174.565562_0-pqr_16384_(all[768]→512)x2→32→1/512-0-pqr_16384_(all[768]→512)x2→32→1.pth'))
chessmodel.cuda()

with open("pqr_eval.csv", "w+") as f:
    f.write("p,q,r\n")

    for _ in tqdm(range(5)):
        X, y = samples.next_batch()
    
        # Expand into floats
        X = expand_batch(X, chessmodel.num_features)

        # Forward pass
        outputs = chessmodel(X)

        # To P,Q,R
        output = outputs.reshape(-1, 3)

        for i in range(output.shape[0]):
            f.write(f"{output[i,0]},{output[i,1]},{output[i,2]}\n")
