import math
import os
import random
import time
from pathlib import Path
import json

from plot.data import experiments

from mt import DATASETS_PATH, DATASET_EVAL_NAME, DATASET_SUMMARY_NAME
from mt import helpers, utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

DATASET = "europarl_fairseq_es-en"
# basepath = f"/Users/salvacarrion/Desktop/euro es-en 32k"
domain, (SRC, TRG) = utils.get_dataset_ids(DATASET)
domain = domain.replace("_fairseq", "").strip()
LIMIT = 100
VOCAB_LANG = TRG

# Create folder
summary_path = os.path.join(DATASETS_PATH, "custom_plots")
Path(summary_path).mkdir(parents=True, exist_ok=True)


def plot_tok_distribution(show_values=False, savepath="."):
    sns.set(font_scale=1.5)  # crazy big
    title = f"{domain} {VOCAB_LANG} (32k vs 64)"

    filename = f"tok_distribution__{title.replace(' ', '_').lower()}__limit{LIMIT}".lower()

    # Build pandas dataframe
    print("Preparing source data...")
    TOK_SIZE = 32000
    basepath = f"/home/scarrion/datasets/scielo/constrained/datasets/bpe.{TOK_SIZE}/{DATASET}/tok/bpe.{TOK_SIZE}"
    with open(os.path.join(basepath, f"vocab.{VOCAB_LANG}"), 'r') as f:
        data_src = [line.strip().split(' ') for line in f.readlines()[:LIMIT]]
        data_src = [(t[0], int(t[1]), "trg") for t in data_src]
        df_src = pd.DataFrame(data=data_src, columns=["subword", "frequency", "split"])

    print("Preparing target data...")
    TOK_SIZE = 64
    basepath = f"/home/scarrion/datasets/scielo/constrained/datasets/bpe.{TOK_SIZE}/{DATASET}/tok/bpe.{TOK_SIZE}"
    with open(os.path.join(basepath, f"vocab.{VOCAB_LANG}"), 'r') as f:
        data_trg = [line.strip().split(' ') for line in f.readlines()[:LIMIT]]
        data_trg = [(t[0], int(t[1]), "trg") for t in data_trg]
        df_trg = pd.DataFrame(data=data_trg, columns=["subword", "frequency", "split"])

    # Define subplots => px, py = w*dpi, h*dpi  # pixels
    print("Plotting...")
    fig, axes = plt.subplots(2, 1, figsize=(12*1.5, 6), dpi=150)  # (nrows, ncols), (W, H), dpi
    plt.ylim([0, 5500000])

    # Image 0
    axes[0].set_title(f"Large subword vocab distribution (~32k tokens)")
    g0 = sns.barplot(ax=axes[0], data=df_src, x="subword", y="frequency")

    # Image 1
    axes[1].set_title(f"Quasi character-level vocab distribution (~300 tokens)")
    g1 = sns.barplot(ax=axes[1], data=df_trg, x="subword", y="frequency")

    # Tweaks
    for gx in [g0, g1]:
        gx.set(xlabel='', ylabel="Frequency")  #
        gx.set_xticklabels(gx.get_xticklabels(), rotation=90)
        gx.tick_params(axis='x', which='major', labelsize=8)
        gx.tick_params(axis='y', which='major', labelsize=8)
        gx.yaxis.set_major_formatter(utils.human_format)
        # gx.set_ylim([0, 5500000])

    # Set yaxis limits + tick frequency
    # g0.yaxis.set_ticks(np.arange(data_src[-1][1], data_src[0][1], 5e5))
    # g1.yaxis.set_ticks(np.arange(data_trg[-1][1], data_trg[0][1], 5e5))
    g0.yaxis.set_ticks(np.arange(0, 5.5*1e6, 5e5))
    g1.yaxis.set_ticks(np.arange(0, 5.5*1e6, 5e5))

    # properties
    plt.tight_layout()

    # Overall title
    # fig.subplots_adjust(top=.88)
    # fig.suptitle(title, fontsize="x-large")

    # Save figure
    plt.savefig(os.path.join(savepath, f"{filename}.pdf"))
    plt.savefig(os.path.join(savepath, f"{filename}.svg"))
    plt.savefig(os.path.join(savepath, f"{filename}.png"))
    print("Figures saved!")

    # Show plot
    plt.show()

    asdsd = 3


if __name__ == "__main__":
    # Plot distribution (leyend BPE)
    plot_tok_distribution(savepath=summary_path)

    print("Done!")
