import os
import subprocess
import os

from mt import utils
from mt import DATASETS_PATH, DATASET_CLEAN_NAME, DATASET_CLEAN_SORTED_NAME, DATASET_TOK_NAME, DATASET_LOGS_NAME, DATASET_CHECKPOINT_NAME


WANDB_PROJECT = "nmt"  # Run "wandb login" in the terminal


def train(datapath):
    subprocess.call(['sh', './scripts/4_train.sh', datapath, WANDB_PROJECT])


if __name__ == "__main__":
    # Get all folders in the root path
    datasets = [os.path.join(DATASETS_PATH, x) for x in ["biological_fairseq_es-en", "merged_fairseq_es-en"]]
    for dataset in datasets:
        domain, (src, trg) = utils.get_dataset_ids(dataset)
        fname_base = f"{domain}_{src}-{trg}"
        print(f"Train fairseq model ({fname_base})...")

        # Train model
        train(dataset)
