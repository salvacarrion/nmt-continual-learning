import os
import pandas as pd

from mt import RAW_PATH
from mt import utils


TR_RAW_FILES = [("es-en-gma-biological.csv", "es-en-gma-health.csv"),
                ("pt-en-gma-biological.csv", "pt-en-gma-health.csv")]
TS_RAW_FILES = [("test-gma-en2es-biological.csv", "test-gma-en2es-health.csv"),
                ("test-gma-en2pt-biological.csv", "test-gma-en2pt-health.csv"),
                ("test-gma-es2en-biological.csv", "test-gma-es2en-health.csv"),
                ("test-gma-pt2en-biological.csv", "test-gma-pt2en-health.csv")]

# Process splits train/test files
for split in ["train", "test"]:

    # Select split to process
    if split == "train":
        print("Processing training files...")
        RAW_FILES = TR_RAW_FILES
        istrain = True

    elif split == "test":
        print("Processing test files...")
        RAW_FILES = TS_RAW_FILES
        istrain = False

    else:
        raise ValueError("Invalid split name")

    # Process raw files
    for fname1, fname2 in RAW_FILES:
        # Read files
        print(f"Reading files... ({fname1} AND {fname2})")
        df_file1 = pd.read_csv(os.path.join(RAW_PATH, fname1))
        df_file2 = pd.read_csv(os.path.join(RAW_PATH, fname2))

        # Add domains
        df_file1["domain"] = "health" if "health" in fname1 else "biological"
        df_file2["domain"] = "health" if "health" in fname2 else "biological"

        # Concat dataframes
        df = pd.concat([df_file1, df_file2])

        # Save data
        save_fname = "-".join(fname1.split('-')[:-1]) + "-merged.csv"
        df.to_csv(os.path.join(RAW_PATH, save_fname), index=False)
        print("File saved!")
        print("")

print("Done!")
