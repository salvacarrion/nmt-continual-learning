import os
from pathlib import Path
import subprocess

from mt import DATASETS_PATH, DATASET_TOK_NAME, DATASET_BPE_NAME, FASTBPE_PATH, VOCAB_SIZE
from mt.preprocess import utils


# Get all folders in the root path
datasets = [os.path.join(DATASETS_PATH, name) for name in os.listdir(DATASETS_PATH) if os.path.isdir(os.path.join(DATASETS_PATH, name))]
# datasets = [os.path.join(DATASETS_PATH, "multi30k_de-en")]
for dataset in datasets:
    domain, (src, trg) = utils.get_dataset_ids(dataset)
    fname_base = f"{domain}_{src}-{trg}"
    print(f"Processing dataset ({fname_base})...")

    # Create path
    savepath = os.path.join(dataset, DATASET_TOK_NAME, DATASET_BPE_NAME)
    Path(savepath).mkdir(parents=True, exist_ok=True)

    # Learn and apply BPE
    subprocess.call(['sh', './scripts/2_learn_bpe.sh', str(VOCAB_SIZE), src, trg, dataset, savepath, FASTBPE_PATH])
    subprocess.call(['sh', './scripts/2_apply_bpe.sh', str(VOCAB_SIZE), src, trg, dataset, savepath, FASTBPE_PATH])
