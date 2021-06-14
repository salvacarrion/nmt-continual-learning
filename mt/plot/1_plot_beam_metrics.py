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

METRIC_KEYS = ['fairseq_bleu', 'sacrebleu_bleu', 'sacrebleu_chrf', 'sacrebleu_ter']
BEAMS = [1, 3, 5, 8, 16, 32]

def get_metrics(datapath, src, trg, model_name, label, train_domain):
    metrics = []
    # Get all folders in the root path
    for test_domain in ["health", "biological", "merged"]:
        # Load metrics to file
        eval_path = os.path.join(datapath, DATASET_EVAL_NAME, model_name, test_domain)
        with open(os.path.join(eval_path, 'beam_metrics.json'), 'r') as f:
            # Load metrics json
            json_metrics = json.load(f)
            key = [k for k in list(json_metrics.keys()) if "beam" in k][0]  # "val"
            row = {}

            # hardcoded
            for metric_key in METRIC_KEYS:
                values = []
                for beam in BEAMS:
                    values.append(json_metrics['beams'][f'beam{beam}'][metric_key])
                row[metric_key] = values

            # Add more data
            row["model_name"] = model_name
            row["label"] = label
            row["train_domain"] = train_domain.title()
            row["test_domain"] = test_domain.title()
            row["lang"] = f"{src}-{trg}"
            metrics.append(row)
    return metrics


def plot_beam_metrics(df, savepath, train_domain, lang_pair, metric=("sacrebleu_bleu", "bleu"), show_values=True, file_title=""):
    metric_id, metric_name = metric

    # Get specific language metrics
    df = df[df.lang == lang_pair]
    # df = df[df.train_domain == train_domain]

    # Re-organize data
    rows = []
    for i, df_row in df.iterrows():
        for j, beam in enumerate(BEAMS):
            row = {"beam_width": beam, "train_domain": df_row["train_domain"], "test_domain": df_row["test_domain"],
                   "domain": f'{df_row["train_domain"]}__{df_row["test_domain"]}'.lower()}
            for metric_key in METRIC_KEYS:
                row[metric_key] = df_row[metric_key][j]
            rows.append(row)
    df_new = pd.DataFrame(rows)
    g = sns.lineplot(data=df_new, x="beam_width", y=metric_id, hue="domain")

    # properties
    g.set(xlabel='Beam width', ylabel=metric_name.upper())
    plt.title(f"Evolution of {metric_name.upper()} | ({lang_pair})")
    plt.legend(loc='upper right', prop={'size': 8})
    plt.tight_layout()

    # Save figure
    plt.savefig(os.path.join(savepath, f"beam_search_{metric_id}_{lang_pair}{file_title}.pdf"))
    plt.savefig(os.path.join(savepath, f"beam_search_{metric_id}_{lang_pair}{file_title}.svg"))
    plt.savefig(os.path.join(savepath, f"beam_search_{metric_id}_{lang_pair}{file_title}.png"))
    print("Figures saved!")

    # Show plot
    plt.show()


if __name__ == "__main__":
    metrics = []

    # Get all folders in the root path
    lang_pair = "es-en"
    train_domain = "health_fairseq".title()
    file_title = ""
    metric = ("fairseq_bleu", "BLEU (Fairseq)")  # (ID, pretty name)
    datasets = [(os.path.join(DATASETS_PATH, x), l) for x, l in [

        # Basic ***********
        ("health_fairseq_es-en", [("checkpoint_best.pt", "Health")]),
        ("biological_fairseq_es-en", [("checkpoint_best.pt", "Biological")]),
        ("merged_fairseq_es-en", [("checkpoint_best.pt", "H+B")]),
        ("health_biological_fairseq_es-en", [("checkpoint_best.pt", "H→B (Voc. domain=H)")]),

    ]]
    for dataset, models in datasets:
        domain, (src, trg) = utils.get_dataset_ids(dataset)
        fname_base = f"{domain}_{src}-{trg}"

        # Train model
        for model_name, label in models:
            print(f"Getting model ({fname_base}; {model_name})...")
            metrics += get_metrics(dataset, src, trg, model_name=model_name, label=label, train_domain=domain)

    # Create folder
    summary_path = os.path.join(DATASETS_PATH, DATASET_SUMMARY_NAME, "metrics")
    Path(summary_path).mkdir(parents=True, exist_ok=True)

    # Save data
    df = pd.DataFrame(metrics)
    # print(df)
    # df.to_csv(os.path.join(summary_path, f"beam_search_data_{train_domain.lower()}-{lang_pair}.csv"))
    # print("Data saved!")

    # Plot metrics
    plot_beam_metrics(df, savepath=summary_path, lang_pair=lang_pair, metric=metric, file_title=file_title, train_domain=train_domain)
    print("Done!")
