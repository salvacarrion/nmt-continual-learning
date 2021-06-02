import math
import os
import random
import time
from pathlib import Path
import json

from mt import DATASETS_PATH, DATASET_EVAL_NAME, DATASET_SUMMARY_NAME
from mt import helpers, utils

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()



def get_metrics(datapath, src, trg, model_name, train_domain):
    metrics = []
    # Get all folders in the root path
    for test_domain in ["health", "biological", "merged"]:
        # Load metrics to file
        eval_path = os.path.join(datapath, DATASET_EVAL_NAME, model_name, test_domain)
        with open(os.path.join(eval_path, 'metrics.json'), 'r') as f:
            # Load metrics json
            row = json.load(f)["val"]

            # Prettify name
            pretty_model_name = model_name.replace("transformer_", "").replace("_best.pt", "").replace("_", " ").strip()  # remove ending ".pt"

            # Add more data
            row["model_name"] = pretty_model_name
            row["train_domain"] = train_domain.title()
            row["test_domain"] = test_domain.title()
            row["lang"] = f"{src}-{trg}"
            metrics.append(row)
    return metrics


def plot_metrics(df_metrics, savepath, lang_pair, metric=("sacrebleu_bleu", "bleu")):
    metric_id, metric_name = metric

    # Get specific language metrics
    df_lang = df_metrics[df_metrics.lang == lang_pair]

    # Draw a nested barplot by species and sex
    g = sns.catplot(data=df_lang, x="model_name", y=metric_id, kind="bar", hue="test_domain", legend=False)
    g.fig.set_size_inches(12, 8)

    # properties
    g.set(xlabel='Models', ylabel=metric_name.upper())
    plt.title(f"{metric_name.upper()} scores in different domains | {lang_pair}")

    g.set_xticklabels(rotation=45, horizontalalignment="center")
    plt.legend(loc='upper right')
    plt.tight_layout()

    # Save figure
    plt.savefig(os.path.join(savepath, f"{metric_id}_scores_{lang_pair}.pdf"))
    plt.savefig(os.path.join(savepath, f"{metric_id}_scores_{lang_pair}.jpg"))
    print("Figure saved!")

    # Show plot
    plt.show()


if __name__ == "__main__":
    metrics = []

    # Get all folders in the root path
    lang_pair = "es-en"
    metric = ("sacrebleu_bleu", "bleu")  # (ID, pretty name)
    datasets = [(os.path.join(DATASETS_PATH, x), l) for x, l in [
        ("health_es-en", ["transformer_health_best.pt"]),
        ("biological_es-en", ["transformer_biological_best.pt"]),
        ("merged_es-en", ["transformer_merged_best.pt"]),
        ("health_biological_inter_es-en", ["transformer_health_biological_inter_a0.0_best.pt", "transformer_health_biological_inter_a0.25_best.pt", "transformer_health_biological_inter_a0.5_best.pt", "transformer_health_biological_inter_a0.75_best.pt"]),
        ("health_biological_lwf_es-en", ["transformer_health_biological_lwf_a0.25_best.pt", "transformer_health_biological_lwf_a0.5_best.pt", "transformer_health_biological_lwf_a0.75_best.pt"]),
    ]]
    for dataset, models in datasets:
        domain, (src, trg) = utils.get_dataset_ids(dataset)
        fname_base = f"{domain}_{src}-{trg}"

        # Train model
        for model_name in models:
            print(f"Getting model ({fname_base}; {model_name})...")
            metrics += get_metrics(dataset, src, trg, model_name=model_name, train_domain=domain)

    # Create folder
    summary_path = os.path.join(DATASETS_PATH, DATASET_SUMMARY_NAME)
    Path(summary_path).mkdir(parents=True, exist_ok=True)

    # Save data
    df = pd.DataFrame(metrics)
    print(df)
    df.to_csv(os.path.join(summary_path, f"test_data.csv"))
    print("Data saved!")

    # Plot metrics
    plot_metrics(df, savepath=summary_path, lang_pair=lang_pair, metric=metric)
    adsd = 3
