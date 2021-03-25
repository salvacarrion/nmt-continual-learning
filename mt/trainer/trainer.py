import os
from mt.trainer.train_model import train_model

BASE_PATH = "/home/salvacarrion/Documents/Programming/Datasets/Scielo/cleaned-constrained/datasets"
FASTBPE_PATH = "/home/salvacarrion/Documents/packages/fastBPE/fast"
#FASTBPE_PATH = "/home/scarrion/packages/fastBPE/fast"
vocab_size = 32000


def get_dataset_ids(dataset_path):
    # Get basename of the path
    basename = os.path.basename(os.path.normpath(dataset_path))

    # Split basename
    domain, langs = basename.split("_")
    src, trg = langs.split("-")
    return domain, (src, trg)


# Get all folders in the root path
datasets = [os.path.join(BASE_PATH, name) for name in os.listdir(BASE_PATH) if os.path.isdir(os.path.join(BASE_PATH, name))]
for dataset in datasets:
    domain, (src, trg) = get_dataset_ids(dataset)
    fname_base = f"{domain}_{src}-{trg}"
    print(f"Training model ({fname_base})...")

    # Train model
    train_model(dataset, src, trg, domain)
