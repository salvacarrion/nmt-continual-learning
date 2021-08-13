import os
import random
from pathlib import Path
import pandas as pd

from mt import DATASETS_PATH, DATASET_RAW_NAME
from mt import utils

SRC_LANG = "es"
TRG_LANG = "en"
val_samples = 5000
test_samples = 5000
shuffle = True
random.seed(1234)


def valid_line(line, min_length=1, max_length=2000, max_diff=200):
    # Min length
    if len(line[0]) < min_length or len(line[1]) < min_length:
        return False

    # Max length
    if len(line[0]) > max_length or len(line[1]) > max_length:
        return False

    # Max diff
    if abs(len(line[0])-len(line[1])) > max_diff:
        return False

    return True

# Get all folders in the root path
datasets = [os.path.join(DATASETS_PATH, "bpe.32000/europarl_fairseq_es-en")]
for dataset in datasets:
    domain, (src, trg) = utils.get_dataset_ids(dataset)
    fname_base = f"{domain}_{src}-{trg}"
    print(f"Processing dataset ({fname_base})...")

    # Create path
    savepath = os.path.join(dataset, "splits")
    Path(savepath).mkdir(parents=True, exist_ok=True)

    # Read files
    src_file = os.path.join(dataset, "raw", f"dataset.{SRC_LANG}")
    with open(src_file, 'r') as f:
        src_lines = f.readlines()
        src_lines = [utils.preprocess_text(l) for l in src_lines]

    trg_file = os.path.join(dataset, "raw", f"dataset.{TRG_LANG}")
    with open(trg_file, 'r') as f:
        trg_lines = f.readlines()
        trg_lines = [utils.preprocess_text(l) for l in trg_lines]

    # Convert to tuple and remove empty lines
    lines = list(zip(src_lines, trg_lines))
    lines_count = len(lines)

    lines = [t for t in lines if valid_line(t)]  # Remove empty lines
    lines_count2 = len(lines)
    print(f"Lines removed: {lines_count-lines_count2}")

    # Shuffle lines
    random.shuffle(lines) if shuffle else None

    # Split
    train, other = lines[:-(val_samples+test_samples)], lines[-(val_samples+test_samples):]
    val = other[:val_samples]
    test = other[val_samples:]

    # Unpack
    train_src, train_trg = zip(*train)
    val_src, val_trg = zip(*val)
    test_src, test_trg = zip(*test)

    # Save split datasets
    with open(os.path.join(savepath, f"train.{SRC_LANG}"), 'w', encoding='utf-8') as f:
        f.writelines([l + '\n' for l in train_src])
    with open(os.path.join(savepath, f"train.{TRG_LANG}"), 'w', encoding='utf-8') as f:
        f.writelines([l + '\n' for l in train_trg])
    with open(os.path.join(savepath, f"val.{SRC_LANG}"), 'w', encoding='utf-8') as f:
        f.writelines([l + '\n' for l in val_src])
    with open(os.path.join(savepath, f"val.{TRG_LANG}"), 'w', encoding='utf-8') as f:
        f.writelines([l + '\n' for l in val_trg])
    with open(os.path.join(savepath, f"test.{SRC_LANG}"), 'w', encoding='utf-8') as f:
        f.writelines([l + '\n' for l in test_src])
    with open(os.path.join(savepath, f"test.{TRG_LANG}"), 'w', encoding='utf-8') as f:
        f.writelines([l + '\n' for l in test_trg])

    print("Done!")


