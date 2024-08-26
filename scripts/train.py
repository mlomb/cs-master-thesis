import argparse
import math
from glob import glob
from tqdm import tqdm
import torch
import wandb
import tempfile

from lib.service import SamplesService, get_feature_set_size
from lib.model import NnueModel
from lib.model import decode_int64_bitset
from lib.serialize import NnueWriter
from lib.puzzles import PuzzleAccuracy
from lib.losses import EvalLoss, PQRLoss


def train(config, use_wandb: bool):
    if config.method == "pqr":
        X_SHAPE = (config.batch_size, 3, 2, math.ceil(config.num_features / 64))
        Y_SHAPE = (config.batch_size, 0)
        INPUTS = glob("/mnt/c/datasets/pqr-1700/*.csv")
        loss_fn = PQRLoss()
    elif config.method == "eval":
        X_SHAPE = (config.batch_size, 2, math.ceil(config.num_features / 64))
        Y_SHAPE = (config.batch_size, 1)
        INPUTS = glob("/mnt/c/datasets/raw/all.plain")
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
        l1_size=config.l1_size,
        l2_size=config.l2_size
    )
    chessmodel.cuda()

    optimizer = torch.optim.Adam(chessmodel.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=config.gamma)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', threshold=0.0001, factor=0.7, patience=30)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.000001, max_lr=0.0002, step_size_up=128)

    @torch.compile
    def train_step(X, y):
        # expand bitset
        # X.shape = [4096, 2, 43]
        X = decode_int64_bitset(X) 
        # X.shape = [4096, 2, 43, 64]
        X = X.reshape(-1, 2, X.shape[-2] * 64)
        # X.shape = [4096, 2, 2752]
        X = X[:, :, :config.num_features] # truncate to the actual number of features
        # X.shape = [4096, 2, 2700]

        # Clear the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = chessmodel(X)

        # Compute the loss
        loss = loss_fn(outputs, y)
        loss.backward()

        # Update the parameters
        optimizer.step()

        # Make sure the weights are clipped
        chessmodel.clip_weights()

        return loss.item()

    # Make sure gradient tracking is on
    chessmodel.train()

    batches_per_epoch = config.epoch_size // config.batch_size

    best_loss = float("inf")
    checkpoint_is_best = True

    for epoch in range(1, config.epochs+1):
        avg_loss = 0.0

        for _ in tqdm(range(batches_per_epoch), desc=f'Epoch {epoch}/{config.epochs}'):
            X, y = samples_service.next_batch()

            loss = train_step(X, y)
            avg_loss += loss

            if math.isnan(avg_loss):
                raise Exception("Loss is NaN, exiting")

        avg_loss /= batches_per_epoch

        # Step the scheduler
        scheduler.step()
        
        checkpoint_is_best = checkpoint_is_best or avg_loss < best_loss
        best_loss = min(best_loss, avg_loss)

        # log metrics to W&B
        metrics = {
            "Train/loss": avg_loss,
            "Train/lr": scheduler._last_lr[0], # get_last_lr()
            "Train/samples": config.batch_size * batches_per_epoch * (epoch + 1),

            "Weight/mean-l1": torch.mean(chessmodel.l1.weight),
            "Weight/mean-l2": torch.mean(chessmodel.l2.weight),
            "Weight/mean-out": torch.mean(chessmodel.output.weight),
        }
        if use_wandb:
            wandb.log(step=epoch, data=metrics)
        else:
            print(f"Step {epoch} - Loss: {avg_loss}, LR: {scheduler._last_lr[0]}")

        with tempfile.NamedTemporaryFile() as tmp:
            tmp.write(NnueWriter(chessmodel, config.feature_set).buf)

            if epoch % config.puzzle_interval == 0 or epoch == 1:
                # run puzzles
                print("tmp.name", tmp.name)
                puzzles_results, puzzles_move_accuracy = puzzles.measure(["../engine/target/release/engine", f"--nn={tmp.name}"])

                # log puzzle metrics
                if use_wandb:
                    wandb.log(step=epoch, data={"Puzzles/moveAccuracy": puzzles_move_accuracy})
                    wandb.log(step=epoch, data={f"Puzzles/{category}": accuracy for category, accuracy in puzzles_results})
                else:
                    print(f"step {epoch} - Puzzles move accuracy: {puzzles_move_accuracy}")

            if epoch % config.checkpoint_interval == 0 or epoch == 1:

                if use_wandb:
                    # store artifact in W&B
                    artifact = wandb.Artifact(f"model_{wandb.run.id}", type="model")
                    artifact.add_file(local_path=tmp.name, name=f"model.nn", policy="mutable")
                    wandb.log_artifact(artifact, aliases=["latest", "best"] if checkpoint_is_best else ["latest"])

                    # reset tag
                    checkpoint_is_best = False
                else:
                    # TODO: make local checkpoint
                    pass

                # build ratings bar chart
                #wandb.log(step=step, data={
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
    parser.add_argument("--feature_set", default="hv", type=str)
    parser.add_argument("--l1_size", default=256, type=int)
    parser.add_argument("--l2_size", default=32, type=int)

    # training
    parser.add_argument("--batch_size", default=16384, type=int, help="Number of samples per minibatch") # 16K
    parser.add_argument("--epoch_size", default=6104 * 16384, type=int, help="Number of samples in one epoch") # 100M
    parser.add_argument("--epochs", default=1024, type=int, help="Number of epochs to train")
    parser.add_argument("--gamma", default=0.992, type=float, help="Multiplier for learning rate decay")
    parser.add_argument("--learning_rate", default=8.75e-4, type=float, help="Initial learning rate")
    parser.add_argument("--method", default="eval", type=str)

    # misc
    parser.add_argument("--checkpoint_interval", default=16, type=int)
    parser.add_argument("--puzzle_interval", default=8, type=int)
    parser.add_argument("--wandb_project", default=None, type=str)

    config = parser.parse_args()

    # compute feature size from feature set
    config.num_features = get_feature_set_size(config.feature_set)

    print(config)

    # torch.set_float32_matmul_precision("high")

    use_wandb = config.wandb_project is not None

    if use_wandb:
        wandb.init(
            project=config.wandb_project,
            job_type="train",
            name=f"{config.method}_{config.batch_size}_{config.feature_set}[{config.num_features}]->{config.l1_size}x2->{config.l2_size}->1",
            config=config
        )
        wandb.define_metric("Train/loss", summary="min")
        wandb.define_metric("Puzzles/moveAccuracy", summary="max")

    train(config, use_wandb)

    if use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
