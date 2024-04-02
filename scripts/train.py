import argparse
import math
from glob import glob
from tqdm import tqdm
import torch
import wandb
import tempfile

from lib.service import SamplesService
from lib.model import NnueModel
from lib.model import decode_int64_bitset
from lib.serialize import NnueWriter
from lib.puzzles import PuzzleAccuracy
from lib.losses import EvalLoss, PQRLoss


def train(config):
    if config.method == "pqr":
        X_SHAPE = (config.batch_size, 3, 2, config.num_features // 64)
        Y_SHAPE = (config.batch_size, 0)
        INPUTS = glob("/mnt/d/datasets/pqr-1700/*.csv")
        loss_fn = PQRLoss()
    elif config.method == "eval":
        X_SHAPE = (config.batch_size, 2, config.num_features // 64)
        Y_SHAPE = (config.batch_size, 1)
        INPUTS = glob("/mnt/d/datasets/eval/*.csv")
        loss_fn = EvalLoss()

    puzzles = PuzzleAccuracy('./data/puzzles.csv')
    samples_service = SamplesService(
        x_shape=X_SHAPE,
        y_shape=Y_SHAPE,
        inputs=INPUTS,
        feature_set=config.feature_set,
        method=config.method
    )
    chessmodel = NnueModel(
        num_features=config.num_features,
        ft_size=config.ft_size,
        l1_size=config.l1_size,
        l2_size=config.l2_size
    )
    chessmodel.cuda()

    optimizer = torch.optim.Adam(chessmodel.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', threshold=0.0001, factor=0.7, patience=15)

    # @torch.compile # 30% speedup
    def train_step(X, y):
        # Clear the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = chessmodel(X)

        # Compute the loss
        loss = loss_fn(outputs, y)
        loss.backward()

        # Update the parameters
        optimizer.step()

        chessmodel.clip_weights()

        return loss

    # Make sure gradient tracking is on
    chessmodel.train()

    for epoch in range(1, config.epochs):
        avg_loss = 0.0

        for _ in tqdm(range(config.batches_per_epoch), desc=f'Epoch {epoch}'):
            X, y = samples_service.next_batch()
        
            # expand bitset
            X = decode_int64_bitset(X)
            X = X.reshape(-1, 2, config.num_features)

            loss = train_step(X, y)
            avg_loss += loss.item()

            if math.isnan(avg_loss):
                raise Exception("Loss is NaN, exiting")

        avg_loss /= config.batches_per_epoch

        # Step the scheduler
        scheduler.step(avg_loss)

        # log metrics to W&B
        wandb.log(step=epoch, data={
            "Train/loss": avg_loss,
            "Train/lr": scheduler._last_lr[0], # get_last_lr()
            "Train/samples": config.batch_size * config.batches_per_epoch * (epoch + 1),

            "Weight/mean-f1": torch.mean(chessmodel.ft.weight),
            "Weight/mean-l1": torch.mean(chessmodel.linear1.weight),
            "Weight/mean-l2": torch.mean(chessmodel.linear2.weight),
            "Weight/mean-out": torch.mean(chessmodel.output.weight),
        })

        with tempfile.NamedTemporaryFile() as tmp:
            tmp.write(NnueWriter(chessmodel, config.feature_set).buf)

            if epoch % config.checkpoint_interval == 0 or epoch == 1:
                # store artifact in W&B
                artifact = wandb.Artifact(wandb.run.id, type="model")
                artifact.add_file(tmp.name, name=f"{epoch}.nn")
                wandb.log_artifact(artifact, aliases=["latest", f"epoch_{epoch}"])

            if epoch % config.puzzle_interval == 0 or epoch == 1:
                # run puzzles
                puzzles_results, puzzles_accuracy = puzzles.measure(["../engine/target/release/engine", f"--nn={tmp.name}"])

                # log puzzle metrics
                wandb.log(step=epoch, data={"Puzzles/accuracy": puzzles_accuracy})
                wandb.log(step=epoch, data={f"Puzzles/{category}": accuracy for category, accuracy in puzzles_results})

                # build ratings bar chart
                #wandb.log(step=epoch, data={
                #    "Puzzles/ratings": wandb.plot.bar(
                #        wandb.Table(
                #            data=sorted([[cat, acc] for cat, acc in puzzles_results if cat.startswith("rating")], 
                #                        key=lambda x: int(x[0].split("rating")[1].split("to")[0])),
                #            columns=["rating", "accuracy"]
                #        ),
                #        label="rating",
                #        value="accuracy",
                #        title="Puzzle accuracy by rating")
                #})

def main():
    parser = argparse.ArgumentParser(description="Train the network")

    # model
    parser.add_argument("--feature-set", default="half-piece", type=str)
    parser.add_argument("--ft-size", default=256, type=int)
    parser.add_argument("--l1-size", default=32, type=int)
    parser.add_argument("--l2-size", default=32, type=int)

    # training
    parser.add_argument("--batch-size", default=4096, type=int)
    parser.add_argument("--batches-per-epoch", default=1000, type=int)
    parser.add_argument("--epochs", default=300, type=int, help="Number of epochs to train for")
    parser.add_argument("--lr", default=0.0015, type=float, help="Initial learning rate")
    parser.add_argument("--method", default="eval", type=str)
    parser.add_argument("--checkpoint-interval", default=10, type=int)
    parser.add_argument("--puzzle-interval", default=30, type=int)

    config = parser.parse_args()

    # compute feature size from feature set
    config.num_features = {
        "half-compact": 192,
        "half-piece": 768,
        "half-king-piece": 40960
    }[config.feature_set]

    print(config)

    wandb.init(
        project="refactor-tests",
        job_type="train",
        name=f"{config.method}_{config.batch_size}_{config.feature_set}[{config.num_features}]->{config.ft_size}x2->{config.l1_size}->{config.l2_size}",
        config=config
    )
    wandb.define_metric("Train/loss", summary="min")
    wandb.define_metric("Puzzles/accuracy", summary="max")

    train(config)

    wandb.finish()


if __name__ == '__main__':
    main()
