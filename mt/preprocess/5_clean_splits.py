import os
from pathlib import Path
import subprocess

from mt import DATASETS_PATH, DATASET_CLEAN_NAME
from mt import utils

# Get all folders in the root path
datasets = [os.path.join(DATASETS_PATH, "bpe.32000/europarl_fairseq_cs-en")]
for dataset in datasets:
    domain, (src, trg) = utils.get_dataset_ids(dataset)
    fname_base = f"{domain}_{src}-{trg}"
    print(f"Processing dataset ({fname_base})...")

    # Create path
    savepath = os.path.join(dataset, DATASET_CLEAN_NAME)
    Path(savepath).mkdir(parents=True, exist_ok=True)

    # Training tokenizer
    subprocess.call(['sh', './scripts/1_tokenize_moses.sh', src, trg, dataset])
