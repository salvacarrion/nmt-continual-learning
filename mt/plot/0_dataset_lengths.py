import math
import os
import random
import time
from pathlib import Path
import json

from mt import DATASETS_PATH, DATASET_TOK_NAME, DATASET_SUMMARY_NAME, DATASET_CLEAN_NAME
from mt import helpers
from mt.preprocess import utils
from mt.trainer.datasets import TranslationDataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

TOK_MODEL = "bpe"
TOK_SIZE = 16000
TOK_FOLDER = f"{TOK_MODEL}.{TOK_SIZE}"
LOWERCASE = False


def get_lengths(datapath, src, trg, domain=None):
    # Load tokenizers
    src_tok, trg_tok = helpers.get_tokenizers(os.path.join(datapath, DATASET_TOK_NAME, TOK_FOLDER), src, trg, tok_model=TOK_MODEL, lower=LOWERCASE)

    # Get datasets
    train_ds = TranslationDataset(os.path.join(datapath, DATASET_CLEAN_NAME), src_tok, trg_tok, "train")
    val_ds = TranslationDataset(os.path.join(datapath, DATASET_CLEAN_NAME), src_tok, trg_tok, "val")
    test_ds = TranslationDataset(os.path.join(datapath, DATASET_CLEAN_NAME), src_tok, trg_tok, "test")

    # Get lengths
    train_lengths = [(len(x["src"]), len(x["trg"])) for x in train_ds]
    val_lengths = [(len(x["src"]), len(x["trg"])) for x in val_ds]
    test_lengths = [(len(x["src"]), len(x["trg"])) for x in test_ds]

    train_src_lengths = [x[0] for x in train_lengths]
    train_trg_lengths = [x[1] for x in train_lengths]
    val_src_lengths = [x[0] for x in val_lengths]
    val_trg_lengths = [x[1] for x in val_lengths]
    test_src_lengths = [x[0] for x in test_lengths]
    test_trg_lengths = [x[1] for x in test_lengths]

    # Prepare lengths
    lengths = {
        "train_src_lengths": train_src_lengths,
        "train_trg_lengths": train_trg_lengths,
        "val_src_lengths": val_src_lengths,
        "val_trg_lengths": val_trg_lengths,
        "test_src_lengths": test_src_lengths,
        "test_trg_lengths": test_trg_lengths,
    }
    return lengths


def plot_length_dist(lengths, lang_pair):
    pairs = lang_pair.split("-")
    for i, src_trg in enumerate(["src", "trg"]):
        data = []
        for split in ["train", "val", "test"]:
            data.append(lengths[f"{split}_{src_trg}_lengths"])

        # Plot histogram
        g = sns.histplot(data=data, kde=True)

        # properties
        g.set(xlabel='Sentence length', ylabel="Frequency")
        plt.title(f"{dataset_name} ({pairs[i]}; {lang_pair})")
        plt.legend(labels=[x.title() for x in ["train", "val", "test"]])
        plt.show()

        # Save figure
        plt.savefig(os.path.join(summary_path_images, f"{fname_base}__{lang_pair}_{pairs[i]}.pdf"))
        plt.savefig(os.path.join(summary_path_images, f"{fname_base}__{lang_pair}_{pairs[i]}.jpg"))
        print("Figure saved!")


if __name__ == "__main__":
    data = []

    # Create folder
    summary_path = os.path.join(DATASETS_PATH, DATASET_SUMMARY_NAME)
    summary_path_images = os.path.join(summary_path, "images")
    Path(summary_path).mkdir(parents=True, exist_ok=True)
    Path(summary_path_images).mkdir(parents=True, exist_ok=True)

    # Get all folders in the root path
    datasets = [("health_es-en", "Health"),
                ("biological_es-en", "Biological"),
                ("merged_es-en", "Health+Biological")]
    datasets = [(os.path.join(DATASETS_PATH, x[0]), x[1]) for x in datasets]
    for dataset, dataset_name in datasets:
        domain, (src, trg) = utils.get_dataset_ids(dataset)
        fname_base = f"{domain}_{src}-{trg}"
        print(f"Getting lengths dataset ({fname_base})...")

        # Get lengths
        lengths = get_lengths(dataset, src, trg, domain=dataset_name)
        row = {"dataset": fname_base, "dataset_name": dataset_name, "domain": domain, "langs": f"{src}-{trg}", "tok_model": TOK_MODEL, "tok_size": TOK_SIZE,

               "train_src_sentences": len(lengths["train_src_lengths"]),
               "train_src_tokens": sum(lengths["train_src_lengths"]),
               "train_src_min_tokens_sentence": min(lengths["train_src_lengths"]),
               "train_src_max_tokens_sentence": max(lengths["train_src_lengths"]),
               "train_trg_sentences": len(lengths["train_trg_lengths"]),
               "train_trg_tokens": sum(lengths["train_trg_lengths"]),
               "train_trg_min_tokens_sentence": min(lengths["train_trg_lengths"]),
               "train_trg_max_tokens_sentence": max(lengths["train_trg_lengths"]),

               "val_src_sentences": len(lengths["val_src_lengths"]),
               "val_src_tokens": sum(lengths["val_src_lengths"]),
               "val_src_min_tokens_sentence": min(lengths["val_src_lengths"]),
               "val_src_max_tokens_sentence": max(lengths["val_src_lengths"]),
               "val_trg_sentences": len(lengths["val_trg_lengths"]),
               "val_trg_tokens": sum(lengths["val_trg_lengths"]),
               "val_trg_min_tokens_sentence": min(lengths["val_trg_lengths"]),
               "val_trg_max_tokens_sentence": max(lengths["val_trg_lengths"]),

               "test_src_sentences": len(lengths["test_src_lengths"]),
               "test_src_tokens": sum(lengths["test_src_lengths"]),
               "test_src_min_tokens_sentence": min(lengths["test_src_lengths"]),
               "test_src_max_tokens_sentence": max(lengths["test_src_lengths"]),
               "test_trg_sentences": len(lengths["test_trg_lengths"]),
               "test_trg_tokens": sum(lengths["test_trg_lengths"]),
               "test_trg_min_tokens_sentence": min(lengths["test_trg_lengths"]),
               "test_trg_max_tokens_sentence": max(lengths["test_trg_lengths"]),
               }
        data.append(row)

        # Plot length distribution
        plot_length_dist(lengths, lang_pair=f"{src}-{trg}")

    # Save data
    df = pd.DataFrame(data)
    print(df)
    df.to_csv(os.path.join(summary_path, f"dataset_lengths.csv"))
    print("Data saved!")

    # Plot senteces/tokens per dataset
    rows = []
    for i, row in df.iterrows():
        for split in ["train", "val", "test"]:
            rows.append({"dataset": row["dataset"], "split": split,
                         f"src_sentences": row[f"{split}_src_sentences"],
                         f"trg_sentences": row[f"{split}_trg_sentences"],
                         f"src_tokens": row[f"{split}_src_tokens"],
                         f"trg_tokens": row[f"{split}_trg_tokens"],

                         })
    df2 = pd.DataFrame(rows)
    for length_type in ["sentences", "tokens"]:
        for src_trg in ["src", "trg"]:
            g = sns.catplot(data=df2, x="dataset", y=f"{src_trg}_{length_type}", kind="bar", hue="split", legend=False)

            # properties
            g.set(xlabel='Datasets', ylabel='Length')
            plt.title(f"Datasets sizes ({length_type.title()} - {src_trg.upper()})")

            g.set_xticklabels(rotation=45, horizontalalignment="center")
            for ax in g.axes.flat:
                ax.yaxis.set_major_formatter(utils.human_format)
            plt.legend(loc='upper right')
            plt.tight_layout()
            plt.show()

            # Save figure
            plt.savefig(os.path.join(summary_path_images, f"split_sizes__{length_type}_{src_trg}.pdf"))
            plt.savefig(os.path.join(summary_path_images, f"split_sizes__{length_type}_{src_trg}.jpg"))
            print("Figure saved!")

