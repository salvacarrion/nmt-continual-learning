import os
import pandas as pd
from pathlib import Path
import numpy as np

from mt import RAW_PATH
from mt.preprocess import utils

SUFFLE = True
CONSTRAINED = True

TR_DATA_PATH = "/home/salvacarrion/Documents/Programming/Datasets/Scielo/originals/scielo-gma/scielo-gma"
TR_RAW_FILES = ["es-en-gma-biological.csv", "es-en-gma-health.csv", "fr-en-gma-health.csv",
                "pt-en-gma-biological.csv", "pt-en-gma-health.csv"]

TS_DATA_PATH = "/home/salvacarrion/Documents/Programming/Datasets/Scielo/originals/testset-gma/testset_gma"
TS_RAW_FILES = ["test-gma-en2es-biological.csv", "test-gma-en2es-health.csv", "test-gma-en2fr-health.csv",
                "test-gma-en2pt-biological.csv", "test-gma-en2pt-health.csv", "test-gma-es2en-biological.csv",
                "test-gma-es2en-health.csv", "test-gma-fr2en-health.csv", "test-gma-pt2en-biological.csv",
                "test-gma-pt2en-health.csv"]


# Create path if doesn't exists
path = Path(RAW_PATH)
path.mkdir(parents=True, exist_ok=True)

# Process splits train/test files
for split in ["train", "test"]:

    # Select split to process
    if split == "train":
        print("Processing training files...")
        DATA_PATH = TR_DATA_PATH
        RAW_FILES = TR_RAW_FILES
        istrain = True

    elif split == "test":
        print("Processing test files...")
        DATA_PATH = TS_DATA_PATH
        RAW_FILES = TS_RAW_FILES
        istrain = False

    else:
        raise ValueError("Invalid split name")

    # Process raw files
    for fname in RAW_FILES:
        # Read file
        print(f"Reading file... ({fname})")
        filename = os.path.join(DATA_PATH, fname)
        df = pd.read_csv(filename)

        # Limit dataset
        domain = utils.get_domain(fname)
        SRC_LANG, TRG_LANG = utils.get_langs(fname, istrain=istrain)

        # Clean dataset (basic)
        total_old = len(df)
        df = utils.preprocess_dataset(df, src_col=SRC_LANG, trg_col=TRG_LANG)

        # Shuffle dataset
        if SUFFLE:
            np.random.seed(123)
            np.random.shuffle(df.values)

        if CONSTRAINED and istrain:
            if domain == "health" and "es" in {SRC_LANG, TRG_LANG}:
                max_size = 123597  # Biological rows
                print(f"Limiting size to {max_size}")
                df = df[:max_size]
            elif domain == "health" and "pt" in {SRC_LANG, TRG_LANG}:
                max_size = 120301  # Biological rows
                print(f"Limiting size to {max_size}")
                df = df[:max_size]

        # Stats
        total_doctypes = df['doctype'].value_counts()
        removed = total_old - len(df)
        print(f"Stats for: {fname} **************************")
        print(f"\t- Documents: {len(set(df['docid']))}")
        print(f"\t- Sentences: {len(df)}")
        print("\t\t- Removed: {} ({:.2f}%)".format(removed, removed / total_old * 100))
        print("\t- Titles/Abstracts: {}/{} ({:.2f}%)".format(total_doctypes['title'], total_doctypes['text'],
                                                             total_doctypes['title'] / total_doctypes['text'] * 100))

        # Save data
        df.to_csv(os.path.join(RAW_PATH, fname), index=False)
        print("File saved!")
        print("")

print("Done!")
