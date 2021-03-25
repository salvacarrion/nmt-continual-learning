import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

from mt.preprocess import RAW_PATH, DATASETS_PATH, DATASET_RAW_NAME
from mt.preprocess import utils

SUFFLE = True
FORCE_BIDIRECTIONAL = False  # (es-en) => [(es-en), (en-es)]


RAW_FILES_TR = ["es-en-gma-biological.csv", "es-en-gma-health.csv", "es-en-gma-merged.csv",
                "pt-en-gma-biological.csv", "pt-en-gma-health.csv", "pt-en-gma-merged.csv"]

# Create path
Path(DATASETS_PATH).mkdir(parents=True, exist_ok=True)


for fname_tr in RAW_FILES_TR:
    # Get domain
    domain = utils.get_domain(fname_tr)

    # Get languages
    SRC_LANG, TRG_LANG = utils.get_langs(fname_tr, istrain=True)

    # Read files
    print(f"Reading files... ({fname_tr})")
    df_tr = pd.read_csv(os.path.join(RAW_PATH, fname_tr))

    # Shuffle dataset
    if SUFFLE:
        np.random.seed(123)
        np.random.shuffle(df_tr.values)

    # Add language pairs
    lang_pairs = [[SRC_LANG, TRG_LANG]]
    if FORCE_BIDIRECTIONAL:
        lang_pairs_reversed = [p[::-1] for p in lang_pairs]
        lang_pairs += lang_pairs_reversed

    # Process language paris
    for src, trg in lang_pairs:
        # Create path
        savepath2 = os.path.join(DATASETS_PATH, f"{domain}_{src}-{trg}", DATASET_RAW_NAME)
        Path(savepath2).mkdir(parents=True, exist_ok=True)

        # Get test
        fname_ts = f"test-gma-{src}2{trg}-{domain}.csv"
        df_ts = pd.read_csv(os.path.join(RAW_PATH, fname_ts))

        # Split train set into train/val
        df_tr_split, df_val_split = train_test_split(df_tr, test_size=len(df_ts))

        # Save splits
        df_tr_split.to_csv(os.path.join(savepath2, f"train_{domain}_{src}-{trg}.csv"), index=False)
        df_val_split.to_csv(os.path.join(savepath2, f"val_{domain}_{src}-{trg}.csv"), index=False)
        df_ts.to_csv(os.path.join(savepath2, f"test_{domain}_{src}-{trg}.csv"), index=False)
        print(f"Splits for {domain}_{src}-{trg} saved!")

    print("Done!")
