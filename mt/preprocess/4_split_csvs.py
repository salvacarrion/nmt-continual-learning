import os
from pathlib import Path

import pandas as pd

from mt.preprocess import DATASETS_PATH, DATASET_RAW_NAME
from mt.preprocess import utils


# Get all folders in the root path
datasets = [os.path.join(DATASETS_PATH, name) for name in os.listdir(DATASETS_PATH) if os.path.isdir(os.path.join(DATASETS_PATH, name))]
for dataset in datasets:
    domain, (src, trg) = utils.get_dataset_ids(dataset)
    fname_base = f"{domain}_{src}-{trg}"
    print(f"Processing dataset ({fname_base})...")

    # Create path
    savepath = os.path.join(dataset, "splits")
    Path(savepath).mkdir(parents=True, exist_ok=True)

    # Load datasets
    print(f"Reading files... ({fname_base})")
    df_train = pd.read_csv(os.path.join(dataset, DATASET_RAW_NAME, "train_" + fname_base + ".csv"))
    df_val = pd.read_csv(os.path.join(dataset, DATASET_RAW_NAME, "val_" + fname_base + ".csv"))
    df_test = pd.read_csv(os.path.join(dataset, DATASET_RAW_NAME, "test_" + fname_base + ".csv"))
    
    # Create splits (csv to txt)
    utils.csv2txt(df_train[src], os.path.join(savepath, f"train.{src}"))
    utils.csv2txt(df_train[trg], os.path.join(savepath, f"train.{trg}"))
    utils.csv2txt(df_val[src], os.path.join(savepath, f"val.{src}"))
    utils.csv2txt(df_val[trg], os.path.join(savepath, f"val.{trg}"))
    utils.csv2txt(df_test[src], os.path.join(savepath, f"test.{src}"))
    utils.csv2txt(df_test[trg], os.path.join(savepath, f"test.{trg}"))
