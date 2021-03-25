import os
from pathlib import Path
from mt.common import lt_src, lt_trg

from mt.preprocess import DATASETS_PATH, DATASET_SPLITS_NAME, VOCAB_SIZE
from mt.preprocess import utils


# Get all folders in the root path
datasets = [os.path.join(DATASETS_PATH, name) for name in os.listdir(DATASETS_PATH) if os.path.isdir(os.path.join(DATASETS_PATH, name))]
for dataset in datasets:
    domain, (src, trg) = utils.get_dataset_ids(dataset)
    fname_base = f"{domain}_{src}-{trg}"
    print(f"Processing dataset ({fname_base})...")

    # Create path
    savepath = os.path.join(dataset, "tok")
    Path(savepath).mkdir(parents=True, exist_ok=True)

    # Training tokenizer
    print(f"Training vocabs...")
    lt_src.train_vocab([os.path.join(DATASETS_PATH, dataset, DATASET_SPLITS_NAME, f"train.{src}")], vocab_size=VOCAB_SIZE)
    lt_trg.train_vocab([os.path.join(DATASETS_PATH, dataset, DATASET_SPLITS_NAME, f"train.{trg}")], vocab_size=VOCAB_SIZE)

    # Save tokenizers
    lt_src.save_vocab(savepath, prefix=f"tok.{src}")
    lt_trg.save_vocab(savepath, prefix=f"tok.{trg}")

    # # Load tokenizer
    # lt_src.load_vocab(os.path.join(BASE_PATH, "tok", f"tok.{SRC}-vocab.json"),
    #                   os.path.join(BASE_PATH, "tok", f"tok.{SRC}-merges.txt"))
    # lt_trg.load_vocab(os.path.join(BASE_PATH, "tok", f"tok.{TRG}-vocab.json"),
    #                   os.path.join(BASE_PATH, "tok", f"tok.{TRG}-merges.txt"))
asdas = 3
