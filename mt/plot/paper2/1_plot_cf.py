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
sns.set(font_scale=2.5)  # crazy big


def get_metrics(datapath, src, trg, model_name, label, train_domain):
    metrics = []
    # Get all folders in the root path
    for test_domain in ["health", "biological", "merged"]:
        # Load metrics to file
        eval_path = os.path.join(datapath, DATASET_EVAL_NAME, model_name, test_domain)
        with open(os.path.join(eval_path, f'beam_metrics.json'), 'r') as f:
            # Load metrics json
            json_metrics = json.load(f)
            row = json_metrics["beams"][BEAM_FOLDER]

            # Add more data
            row["model_name"] = model_name
            row["label"] = label.replace("small; ", "").replace("VD", "Voc. domain")
            row["train_domain"] = train_domain.lower().replace("_fairseq", "").lower()
            row["test_domain"] = f"{test_domain.title()}"
            row["vocab_domain"] = row['train_domain'].split('_')[-1][1:].title()
            row["lang"] = f"{src}-{trg}"
            row["vocab_size"] = VOCAB_STR
            metrics.append(row)
    return metrics


def plot_metrics(df_metrics, savepath, lang_pair, metric=("sacrebleu_bleu", "bleu"), show_values=True, vocab=None, tok_size=None):
    metric_id, metric_name = metric

    # Get specific language metrics
    df_metrics = df_metrics[df_metrics.lang == lang_pair]
    df_metrics = df_metrics[df_metrics.test_domain != "Merged"]  # To remove columns from the chart

    # Define subplots => px, py = w*dpi, h*dpi  # pixels
    size = (1, 1)  # H, W
    scales = (7, 18)  # H, W

    # Draw a nested barplot by species and sex
    g = sns.catplot(data=df_metrics, x="label", y=metric_id, kind="bar", hue="test_domain", legend=False)
    g.fig.set_size_inches(scales[1]*size[1], scales[0]*size[0])

    # Add values
    if show_values:
        ax = g.facet_axis(0, 0)
        ax.set_ylim([0, 50])
        ax.axes.get_xaxis().get_label().set_visible(False)

        for c in ax.containers:
            labels = [f"{float(v.get_height()):.1f}" for v in c]
            ax.bar_label(c, labels=labels, label_type='edge')

    # properties
    g.set(xlabel='', ylabel=metric_name.upper())
    g.set_xticklabels(rotation=0, horizontalalignment="center")
    if tok_size == 32000:
        plt.title(f"Voc. size: ~32k tok | Voc. domain: {vocab.title()}")
    else:
        plt.title(f"Voc size: ~350 tok | Voc. domain: {vocab.title()}")
    plt.ylim([0, 50])
    plt.legend(loc='lower right')
    plt.tight_layout()

    # Save figure
    plt.savefig(os.path.join(savepath, f"{metric_id}_{tok_size}_v{vocab}.pdf"), dpi=150)
    plt.savefig(os.path.join(savepath, f"{metric_id}_{tok_size}_v{vocab}.svg"), dpi=150)
    plt.savefig(os.path.join(savepath, f"{metric_id}_{tok_size}_v{vocab}.png"), dpi=150)
    print("Figures saved!")

    # Show plot
    plt.show()
    asd = 44


if __name__ == "__main__":
    # Create folder
    summary_path = os.path.join(DATASETS_PATH, "custom_plots", "cf")
    Path(summary_path).mkdir(parents=True, exist_ok=True)

    BEAM_FOLDER = "beam5"
    METRIC = "bleu"
    lang_pair = "es-en"
    TOK_SIZES = [64, 32000]
    TOK_MODEL = "bpe"
    metric = ("sacrebleu_bleu", "bleu")  # (ID, pretty name)

    for TOK_SIZE in TOK_SIZES:
        TOK_FOLDER = f"{TOK_MODEL}.{TOK_SIZE}"
        VOCAB_STR = str(TOK_SIZE)[:-3] + "k" if len(str(TOK_SIZE)) > 3 else str(TOK_SIZE)

        for vocab in ["health", "biological"]:
            metrics = []

            datasets = [(os.path.join(DATASETS_PATH, TOK_FOLDER, x), l) for x, l in [
                (f"health_fairseq_v{vocab}_es-en", [("checkpoint_best.pt", f"Health\n(small; VD={vocab[0].upper()})")]),
                (f"biological_fairseq_v{vocab}_es-en", [("checkpoint_best.pt", f"Biological\n(small; VD={vocab[0].upper()})")]),
                # (f"merged_fairseq_v{vocab}_es-en", [("checkpoint_best.pt", f"Merged\n(small; VD={vocab[0].upper()})")]),
                (f"health_biological_fairseq_v{vocab}_es-en", [("checkpoint_best.pt", f"H→B\n(small; VD={vocab[0].upper()})")]),

            ]]

            for dataset, models in datasets:
                    print(f"Setting vocab ({vocab})...")
                    domain, (src, trg) = utils.get_dataset_ids(dataset)
                    fname_base = f"{domain}_{src}-{trg}"

                    # Train model
                    for model_name, label in models:
                        print(f"Getting model ({fname_base}; {model_name})...")
                        metrics += get_metrics(dataset, src, trg, model_name=model_name, label=label, train_domain=domain)

            # Save data
            df = pd.DataFrame(metrics)
            df.to_csv(os.path.join(summary_path, f"data_{TOK_SIZE}_v{vocab}.csv"))
            print("Data saved!")

            # Plot metrics
            plot_metrics(df, savepath=summary_path, lang_pair=lang_pair, metric=metric, vocab=vocab, tok_size=TOK_SIZE)
            print("Done!")
