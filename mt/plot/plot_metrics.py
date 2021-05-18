import math
import os
import random
import time
from pathlib import Path
import json

from mt import DATASETS_PATH, DATASET_EVAL_NAME, DATASET_SUMMARY_NAME
from mt import helpers
from mt.preprocess import utils

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


MODEL_NAME = "transformer"


def get_metrics(datapath, src, trg, model_name, domain=None, smart_batch=False):
    metrics = []
    # Get all folders in the root path
    for test_domain in ["health", "biological", "merged"]:
        # Load metrics to file
        eval_path = os.path.join(datapath, DATASET_EVAL_NAME, test_domain)
        with open(os.path.join(eval_path, 'metrics.json'), 'r') as f:
            row = json.load(f)["val"]
            row["model_domain"] = domain
            row["test_domain"] = test_domain.title()
            row["lang"] = f"{src}-{trg}"
            metrics.append(row)

    return metrics


def plot_metrics(df_metrics, savepath, lang_pair, column="bleu"):
    # Get specific language metrics
    df_lang = df_metrics[df_metrics.lang == lang_pair]

    # Draw a nested barplot by species and sex
    g = sns.catplot(data=df_lang, x="model_domain", y=column, kind="bar", hue="test_domain", legend=False)
    g.fig.set_size_inches(12, 8)

    # properties
    g.set(xlabel='Models', ylabel='BLEU')
    plt.title(f"{column.upper()} scores in different domains | {lang_pair}")

    g.set_xticklabels(rotation=0, horizontalalignment="center")
    plt.legend(loc='upper right')
    plt.tight_layout()

    # Save figure
    plt.savefig(os.path.join(savepath, f"{column}_scores_{lang_pair}.pdf"))
    plt.savefig(os.path.join(savepath, f"{column}_scores_{lang_pair}.jpg"))
    print("Figure saved!")

    # Show plot
    plt.show()


if __name__ == "__main__":
    metrics = []

    # Get all folders in the root path
    lang_pair = "es-en"
    datasets = [("health_es-en", "Health"),
                ("biological_es-en", "Biological"),
                ("merged_es-en", "Health+Biological")]
    datasets = [(os.path.join(DATASETS_PATH, x[0]), x[1]) for x in datasets]
    for dataset, dataset_name in datasets:
        domain, (src, trg) = utils.get_dataset_ids(dataset)
        fname_base = f"{domain}_{src}-{trg}"
        print(f"Plotting model ({fname_base})...")

        # Train model
        metrics += get_metrics(dataset, src, trg, model_name=MODEL_NAME, domain=dataset_name)

    # Create folder
    summary_path = os.path.join(DATASETS_PATH, DATASET_SUMMARY_NAME)
    Path(summary_path).mkdir(parents=True, exist_ok=True)

    # Save data
    df = pd.DataFrame(metrics, columns=["model_domain", "test_domain", "lang", "bleu", "loss", "ppl"])
    print(df)
    df.to_csv(os.path.join(summary_path, f"test_data.csv"))
    print("Data saved!")

    # Plot metrics
    plot_metrics(df, savepath=summary_path, lang_pair=lang_pair)
    adsd = 3
