import os
from pathlib import Path
import subprocess

from mt import DATASETS_PATH, DATASET_CLEAN_NAME, DATASET_CLEAN_SORTED_NAME
from mt import utils

# Get all folders in the root path
datasets = [os.path.join(DATASETS_PATH, name) for name in os.listdir(DATASETS_PATH) if os.path.isdir(os.path.join(DATASETS_PATH, name))]
# datasets = [os.path.join(DATASETS_PATH, "multi30k_de-en")]
for dataset in datasets:
    domain, (src, trg) = utils.get_dataset_ids(dataset)
    fname_base = f"{domain}_{src}-{trg}"
    print(f"Processing dataset ({fname_base})...")

    # Create path
    savepath = os.path.join(dataset, DATASET_CLEAN_SORTED_NAME)
    Path(savepath).mkdir(parents=True, exist_ok=True)

    # Training tokenizer
    for split in ["train", "val", "test"]:
        # Open clean file
        with open(os.path.join(dataset, DATASET_CLEAN_NAME, f"{split}.{src}"), 'r') as f:
            lines_src = f.readlines()
        with open(os.path.join(dataset, DATASET_CLEAN_NAME, f"{split}.{trg}"), 'r') as f:
            lines_trg = f.readlines()

        # Sorted positions
        idxs = sorted(range(len(lines_src)), key=lambda x: len(lines_src[x]))
        sorted_lines_src = [lines_src[x] for x in idxs]
        sorted_lines_trg = [lines_trg[x] for x in idxs]

        # Save sorted lines
        with open(os.path.join(dataset, DATASET_CLEAN_SORTED_NAME, f"{split}.{src}"), 'w') as f:
            f.writelines(sorted_lines_src)
        with open(os.path.join(dataset, DATASET_CLEAN_SORTED_NAME, f"{split}.{trg}"), 'w') as f:
            f.writelines(sorted_lines_trg)
        print("File saved!")
    asd = 3
