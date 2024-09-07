import sys
sys.path.append("lib")

import argparse
from tqdm import tqdm
import torch
import wandb
import tempfile
import math

from lib.batch_loader import BatchLoader, get_feature_set_size
from lib.model import NnueModel, expand_batch
from lib.serialize import NnueWriter
from lib.puzzles import Puzzles
from lib.losses import EvalLoss, PQRLoss
from lib.paths import DEFAULT_DATASET, ENGINE_BIN
from lib.games import Engine, measure_perf_diff


def train(config: dict, use_wandb: bool):
    VALIDATION_BYTES = 100_000_000  # ~ 4.5M samples

    # datasets
    train_samples = BatchLoader(
        batch_size=config.batch_size,
        batch_threads=8,
        input=config.dataset,
        input_offset=VALIDATION_BYTES,
        input_loop=True,  # loop infinitely
        feature_set=config.feature_set,
        method=config.method,
        random_skipping=0.3,
    )
    val_samples = BatchLoader(
        batch_size=config.batch_size,
        batch_threads=8,
        input=config.dataset,
        input_length=VALIDATION_BYTES,
        feature_set=config.feature_set,
        method=config.method,
    )

    if config.method == "pqr":
        loss_fn = PQRLoss()
    elif config.method == "eval":
        loss_fn = EvalLoss()

    # model
    chessmodel = NnueModel(
        num_features=config.num_features,
        l1_size=config.l1_size,
        l2_size=config.l2_size
    )
    chessmodel.cuda()

    optimizer = torch.optim.Adam(chessmodel.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=config.gamma)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', threshold=0.0001, factor=0.7, patience=30)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.000001, max_lr=0.0002, step_size_up=128)

    batches_per_epoch = config.epoch_size // config.batch_size
    
    def forward_loss(X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Takes a compressed batch and computes the loss
        """
        # Expand into floats
        X = expand_batch(X, config.num_features)

        # Forward pass
        outputs = chessmodel(X)

        # Compute the loss
        loss = loss_fn(outputs, y)

        return loss

    @torch.compile
    def train_pass(X: torch.Tensor, y: torch.Tensor) -> float:
        """
        Train a single batch
        Returns the loss for the batch
        """
        # Clear the gradients
        optimizer.zero_grad()

        # Compute loss
        loss = forward_loss(X, y)
        loss.backward()

        # Update the parameters
        optimizer.step()

        # Make sure the weights are clipped
        chessmodel.clip_weights()

        return loss.item()

    def run_epoch():
        """
        Trains for one epoch.
        Returns train and validation loss
        """
        train_sum = 0.0
        val_sum = 0.0
        val_count = 0

        # ======================================
        #                 TRAIN
        # ======================================
        # Make sure gradient tracking is on
        chessmodel.train()

        for _ in tqdm(range(batches_per_epoch), desc=f'Epoch {epoch}/{config.epochs}'):
            X, y = train_samples.next_batch()

            train_sum += train_pass(X, y)

            if math.isnan(train_sum):
                raise ValueError("NaN detected in training loss")

        # Step the scheduler
        scheduler.step()

        # ======================================
        #              VALIDATION
        # ======================================
        # Make sure we are not tracking gradients (it's faster)
        chessmodel.eval()

        while True:
            batch = val_samples.next_batch()
            if batch is None:
                break

            X, y = batch
            val_sum += forward_loss(X, y).item()
            val_count += 1

        return (
            train_sum / batches_per_epoch,
            val_sum / val_count
        )

    best_loss = float("inf")
    checkpoint_is_best = True

    for epoch in range(1, config.epochs+1):
        train_loss, val_loss = run_epoch()

        checkpoint_is_best = checkpoint_is_best or val_loss < best_loss
        best_loss = min(best_loss, val_loss)

        # log metrics to W&B
        metrics = {
            "Train/train_loss": train_loss,
            "Train/val_loss": val_loss,
            "Train/lr": scheduler._last_lr[0], # get_last_lr()
            "Train/samples": config.batch_size * batches_per_epoch * (epoch + 1),

            "Weight/mean-l1": torch.mean(chessmodel.l1.weight),
            "Weight/mean-l2": torch.mean(chessmodel.l2.weight),
            "Weight/mean-out": torch.mean(chessmodel.output.weight),
        }

        if use_wandb:
            wandb.log(step=epoch, data=metrics)
        else:
            print(f"Epoch {epoch} - loss: {train_loss}, val_loss: {val_loss}, lr: {scheduler._last_lr[0]}")

        with tempfile.NamedTemporaryFile() as tmp:
            tmp.write(NnueWriter(chessmodel, config.feature_set).buf)

            # ======================================
            #              CHECKPOINTS
            # ======================================
            if epoch % config.checkpoint_interval == 0 or epoch == 1:

                if use_wandb:
                    # store artifact in W&B
                    artifact = wandb.Artifact(f"model_{wandb.run.id}", type="model")
                    artifact.add_file(local_path=tmp.name, name=f"model.nn", policy="mutable")
                    wandb.log_artifact(artifact, aliases=["latest", "best"] if checkpoint_is_best else ["latest"])

                    # reset tag
                    checkpoint_is_best = False
                else:
                    # make local checkpoint
                    # torch.save(chessmodel.state_dict(), f"checkpoints/{epoch}.pth")
                    pass

            # ======================================
            #                PUZZLES
            # ======================================
            if epoch % config.puzzle_interval == 0 or epoch == 1:
                # run puzzles
                puzzles_results, puzzles_move_accuracy = Puzzles().measure([ENGINE_BIN, f"--nn={tmp.name}"])

                # log puzzle metrics
                if use_wandb:
                    wandb.log(step=epoch, data={"Puzzles/moveAccuracy": puzzles_move_accuracy})
                    wandb.log(step=epoch, data={f"Puzzles/{category}": accuracy for category, accuracy in puzzles_results})
                else:
                    print(f"Epoch {epoch} - Puzzles move accuracy: {puzzles_move_accuracy}")

            # ======================================
            #                  PERF
            # ======================================
            if config.perf_interval > 0 and (epoch % config.perf_interval == 0 or epoch == 1):
                elo_diff, error, points = measure_perf_diff(
                    engine1=Engine(name="engine", cmd=ENGINE_BIN, args=[f"--nn={tmp.name}"]),
                    n=200,
                )

                if use_wandb:
                    wandb.log(step=epoch, data={"Perf/elo_diff": elo_diff, "Perf/elo_err": error, "Perf/points": points})
                else:
                    print(f"Epoch {epoch} - ELO: {elo_diff} ± {error} Points: {points}")


def main():
    parser = argparse.ArgumentParser(description="Train the network")

    # model
    parser.add_argument("--feature_set", default="hv", type=str)
    parser.add_argument("--l1_size", default=256, type=int)
    parser.add_argument("--l2_size", default=32, type=int)

    # training
    parser.add_argument("--method", default="eval", type=str)
    parser.add_argument("--dataset", default=DEFAULT_DATASET, type=str, help="Path to the .plain dataset. The first 100MB are used as validation set")

    # hyperparams
    parser.add_argument("--batch_size", default=16384, type=int, help="Number of samples per minibatch") # 16K
    parser.add_argument("--epoch_size", default=6104 * 16384, type=int, help="Number of samples in one epoch") # 100M
    parser.add_argument("--epochs", default=1024, type=int, help="Number of epochs to train")
    parser.add_argument("--learning_rate", default=8.75e-4, type=float, help="Initial learning rate")
    parser.add_argument("--gamma", default=0.994, type=float, help="Multiplier for learning rate decay")

    # misc
    parser.add_argument("--checkpoint_interval", default=16, type=int)
    parser.add_argument("--puzzle_interval", default=8, type=int)
    parser.add_argument("--perf_interval", default=0, type=int)

    # wandb
    parser.add_argument("--wandb", default=None, type=str, help="wandb project name")
    parser.add_argument("--notes", default=None, type=str, help="wandb run notes, a short description of the run")

    config = parser.parse_args()

    # compute feature size from feature set
    config.num_features = get_feature_set_size(config.feature_set)

    print(config)

    # TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled.
    torch.set_float32_matmul_precision("high")

    use_wandb = config.wandb is not None

    if use_wandb:
        wandb.init(
            project=config.wandb,
            job_type="train",
            name=f"{config.method}_{config.batch_size}_({config.feature_set}[{config.num_features}]->{config.l1_size})x2->{config.l2_size}->1",
            notes=config.notes,
            config=config
        )
        wandb.define_metric("Train/train_loss", summary="min")
        wandb.define_metric("Train/val_loss", summary="min")
        wandb.define_metric("Perf/elo_diff", summary="max")
        wandb.define_metric("Perf/points", summary="max")
        wandb.define_metric("Puzzles/moveAccuracy", summary="max")

    train(config, use_wandb)

    if use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
