import argparse
import os
import shutil
from tqdm import tqdm
import wandb

def main():
    parser = argparse.ArgumentParser(description="Downloads the latest model from a wandb sweep")

    parser.add_argument("--wandb-project", default="mlomb/cs-master-thesis", type=str)
    parser.add_argument("--wandb-sweep", default=None, type=str)

    parser.add_argument("--output-dir", default="models", type=str)

    config = parser.parse_args()

    if config.wandb_sweep is None:
        print("No sweep ID specified")
        return

    api = wandb.Api()

    runs = api.runs(
        config.wandb_project,
        filters={"Sweep": config.wandb_sweep}
    )

    print(f"Found {len(runs)} runs")

    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    for run in tqdm(runs):
        artifact = api.artifact(f"mlomb/cs-master-thesis/model_{run.id}:best")
        path = artifact.download()

        # list single file and copy to output directory
        files = os.listdir(path)
        assert len(files) == 1

        def fix_name(name):
            # replace:
            # [ with (
            # ] with )
            # > with _
            return name.replace("[", "(").replace("]", ")").replace(">", "_")

        model_file = os.listdir(path)[0]
        shutil.copyfile(os.path.join(path, model_file), os.path.join(config.output_dir, run.id + "_" + fix_name(run.name) + ".nn"))

    print("Done")

if __name__ == '__main__':
    main()
