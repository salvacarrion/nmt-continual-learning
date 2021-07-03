import os
import subprocess
import os

from mt import utils
from mt import DATASETS_PATH, DATASET_CLEAN_NAME, DATASET_CLEAN_SORTED_NAME, DATASET_TOK_NAME, DATASET_LOGS_NAME, DATASET_CHECKPOINT_NAME


WANDB_PROJECT = "nmt"  # Run "wandb login" in the terminal
USE_LARGE_MODEL = False

TOK_MODEL = "bpe"  # wt
TOK_SIZE = 2000
TOK_FOLDER = f"{TOK_MODEL}.{TOK_SIZE}"


def train(datapath):
    script = "4_train_large.sh" if USE_LARGE_MODEL else "4_train_small.sh"
    subprocess.call(['sh', f'./scripts/{script}', datapath, WANDB_PROJECT])


if __name__ == "__main__":
    # Get all folders in the root path
    datasets = [os.path.join(DATASETS_PATH, TOK_FOLDER, x) for x in [
        # "health_fairseq_vhealth_es-en",
        # "health_fairseq_vbiological_es-en",
        # "health_fairseq_vmerged_es-en",
        #
        # "biological_fairseq_vhealth_es-en",
        # "biological_fairseq_vbiological_es-en",
        # "biological_fairseq_vmerged_es-en",
        #
        # "merged_fairseq_vhealth_es-en",
        # "merged_fairseq_vbiological_es-en",
        # "merged_fairseq_vmerged_es-en",

        "health_biological_fairseq_vhealth_es-en",
        "health_biological_fairseq_vbiological_es-en",
        "health_biological_fairseq_vmerged_es-en",

        # "health_fairseq_large_vhealth_es-en",
        # "biological_fairseq_large_vbiological_es-en",
        # "merged_fairseq_large_vmerged_es-en",
        # "health_biological_fairseq_large_vhealth_es-en",
    ]]
    for dataset in datasets:
        domain, (src, trg) = utils.get_dataset_ids(dataset)
        fname_base = f"{domain}_{src}-{trg}"
        print(f"Train fairseq model ({fname_base})...")

        # Train model
        train(dataset)
