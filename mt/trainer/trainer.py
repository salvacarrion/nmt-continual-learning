import os
from mt.trainer.train_model import train_model

from mt.preprocess import DATASETS_PATH
from mt.preprocess import utils


# Get all folders in the root path
#datasets = [os.path.join(DATASETS_PATH, name) for name in os.listdir(DATASETS_PATH) if os.path.isdir(os.path.join(DATASETS_PATH, name))]
datasets = [os.path.join(DATASETS_PATH, "tmp|health_es-en")]
for dataset in datasets:
    domain, (src, trg) = utils.get_dataset_ids(dataset)
    fname_base = f"{domain}_{src}-{trg}"
    print(f"Training model ({fname_base})...")

    # Train model
    train_model(dataset, src, trg, domain)
