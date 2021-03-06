import os
from pathlib import Path
import subprocess

from mt import DATASETS_PATH, DATASET_TOK_NAME, DATASET_BPE_NAME, FASTBPE_PATH
from mt import utils

VOCAB_SIZE = 32000
TOK_FOLDER = f"{DATASET_BPE_NAME}.{VOCAB_SIZE}"
SAVE_VOCABS = False  # Always "False" since we don't want to modify the vocabulary

# Get all folders in the root path
# datasets = [os.path.join(DATASETS_PATH, name) for name in os.listdir(DATASETS_PATH) if os.path.isdir(os.path.join(DATASETS_PATH, name))]
datasets = [os.path.join(DATASETS_PATH, "health_biological_fairseq_es-en")]
for dataset in datasets:
    domain, (src, trg) = utils.get_dataset_ids(dataset)
    fname_base = f"{domain}_{src}-{trg}"
    print(f"Processing dataset ({fname_base})...")

    # Create path
    savepath = os.path.join(dataset, DATASET_TOK_NAME, TOK_FOLDER)
    Path(savepath).mkdir(parents=True, exist_ok=True)

    # Learn and apply BPE
    subprocess.call(['sh', './scripts/2_apply_bpe.sh', str(VOCAB_SIZE), src, trg, dataset, savepath, FASTBPE_PATH, "true" if SAVE_VOCABS else "false"])

