import math
import os
import random
import time
from pathlib import Path
import json

from mt import DATASETS_PATH, DATASET_TOK_NAME, DATASET_SUMMARY_NAME
from mt import helpers, utils

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

TOK_MODEL = "bpe"
TOK_SIZE = 8000
TOK_FOLDER = f"{TOK_MODEL}.{TOK_SIZE}"
DOMAINS = ["health", "biological", "merged"]
VOCAB_STR = str(TOK_SIZE)[:-3] + "k"
LANG_PAIR = "es-en"
METRIC = "iou"


def get_vocabs(lang_pair):
    vocabs = {}

    # Get all folders in the root path
    for domain in DOMAINS:
        vocabs[domain] = {}

        for lang in lang_pair.split("-"):
            with open(os.path.join(DATASETS_PATH, TOK_FOLDER, f"{domain}_fairseq_v{domain}_{lang_pair}", DATASET_TOK_NAME, TOK_FOLDER, f"vocab.{lang}"), 'r') as f:
                vocabs[domain][lang] = {w.split(' ')[0].strip() for w in f.read().strip().split('\n')}

    return vocabs


def compute_overlap(vocabs, lang_pair):
    rows = []

    # Compute Overlap
    for lang in lang_pair.split("-"):
        for domain1 in DOMAINS:
            vocab1 = vocabs[domain1][lang]

            for domain2 in DOMAINS:
                vocab2 = vocabs[domain2][lang]

                # Compute overlap
                intersection = len(vocab1.intersection(vocab2))
                union = len(vocab1.union(vocab2))
                iou = intersection / union  # Intersection over union
                iov = intersection / len(vocab1)  # Intersection over vocab

                # Add row
                row = {"dataset": lang_pair, "lang": lang, "domain1": domain1, "domain2": domain2,
                       "iov": iov, "iou": iou, "intersection": intersection, "union": union}
                rows.append(row)

    # Create pandas
    df = pd.DataFrame(data=rows)
    return df


def plot_heat_map(df, lang_pair, metric, savepath):
    for lang in lang_pair.split("-"):
        plt.figure()
        data = np.zeros((3, 3))
        for i, domain1 in enumerate(DOMAINS):
            for j, domain2 in enumerate(DOMAINS):
                mask = (df["dataset"] == lang_pair) & (df["lang"] == lang) & (df["domain1"] == domain1) & (
                            df["domain2"] == domain2)
                row = df[mask]
                value = float(row[metric].values[0])
                data[i, j] = value

        g = sns.heatmap(data, annot=True)
        g.set_xticklabels([x.title() for x in DOMAINS], ha='center', minor=False)
        g.set_yticklabels([x.title() for x in DOMAINS], va='center', minor=False)

        # properties
        plt.title(f"Overlapping ratios ({metric.upper()}) | {VOCAB_STR} | {lang} ({lang_pair})")
        plt.tight_layout()

        # Save figure
        plt.savefig(os.path.join(savepath, f"overlapping_{VOCAB_STR}_{metric}_{lang}.pdf"))
        plt.savefig(os.path.join(savepath, f"overlapping_{VOCAB_STR}_{metric}_{lang}.svg"))
        plt.savefig(os.path.join(savepath, f"overlapping_{VOCAB_STR}_{metric}_{lang}.png"))
        print("Figures saved!")

        # Show plot
        plt.show()


if __name__ == "__main__":
    file_title = "__" + "vocab_overlap"

    # Create folder
    summary_path = os.path.join(DATASETS_PATH, TOK_FOLDER, DATASET_SUMMARY_NAME, "vocab_overlap")
    Path(summary_path).mkdir(parents=True, exist_ok=True)

    # Get vocabs
    vocabs = get_vocabs(LANG_PAIR)

    # Compute overlap and print
    df = compute_overlap(vocabs, LANG_PAIR)
    print(df)

    # Save file
    df.to_csv(os.path.join(summary_path, f"overlapping_{VOCAB_STR}_{LANG_PAIR}.csv"), index=False)
    print("File saved!")

    # Plot heatmap
    plot_heat_map(df, LANG_PAIR, METRIC, summary_path)
    print("Done!")
