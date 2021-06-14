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


def get_metrics(datapath, src, trg, model_name, label, train_domain):
    metrics = []
    # Get all folders in the root path
    for test_domain in ["health", "biological", "merged"]:
        # Load metrics to file
        eval_path = os.path.join(datapath, DATASET_EVAL_NAME, model_name, test_domain)
        with open(os.path.join(eval_path, 'metrics.json'), 'r') as f:
            # Load metrics json
            json_metrics = json.load(f)
            key = [k for k in list(json_metrics.keys()) if "beam" in k][0]  # "val"
            row = json_metrics[key]

            # Add more data
            row["model_name"] = model_name
            row["label"] = label
            row["train_domain"] = train_domain.title()
            row["test_domain"] = test_domain.title()
            row["lang"] = f"{src}-{trg}"
            metrics.append(row)
    return metrics


def plot_metrics(df_metrics, savepath, lang_pair, metric=("sacrebleu_bleu", "bleu"), show_values=True, file_title=""):
    metric_id, metric_name = metric

    # Get specific language metrics
    df_lang = df_metrics[df_metrics.lang == lang_pair]

    # Draw a nested barplot by species and sex
    g = sns.catplot(data=df_lang, x="label", y=metric_id, kind="bar", hue="test_domain", legend=False)
    g.fig.set_size_inches(12, 8)

    # Add values
    if show_values:
        ax = g.facet_axis(0, 0)
        for c in ax.containers:
            labels = [f"{float(v.get_height()):.1f}" for v in c]
            ax.bar_label(c, labels=labels, label_type='edge', fontsize=8)

    # properties
    g.set(xlabel='Models', ylabel=metric_name.upper())
    plt.title(f"{metric_name.upper()} scores in different domains | {lang_pair}")

    g.set_xticklabels(rotation=0, horizontalalignment="center")
    plt.legend(loc='lower right')
    plt.tight_layout()

    # Save figure
    plt.savefig(os.path.join(savepath, f"{metric_id}_scores_{lang_pair}{file_title}.pdf"))
    plt.savefig(os.path.join(savepath, f"{metric_id}_scores_{lang_pair}{file_title}.svg"))
    plt.savefig(os.path.join(savepath, f"{metric_id}_scores_{lang_pair}{file_title}.png"))
    print("Figures saved!")

    # Show plot
    plt.show()


if __name__ == "__main__":
    metrics = []

    # Get all folders in the root path
    lang_pair = "es-en"
    file_title = "__" + "hbm_basic"
    metric = ("sacrebleu_bleu", "bleu")  # (ID, pretty name)
    datasets = [(os.path.join(DATASETS_PATH, x), l) for x, l in [

        # Basic ***********
        ("health_fairseq_es-en", [("checkpoint_best.pt", "Health (Fairseq; small)")]),
        ("biological_fairseq_es-en", [("checkpoint_best.pt", "Biological (Fairseq; small)")]),
        ("merged_fairseq_es-en", [("checkpoint_best.pt", "H+B (Fairseq; small)")]),

        # Model size ***********
        # ("health_fairseq_es-en", [("checkpoint_best.pt", "Health (Fairseq; small)")]),
        # ("health_fairseq_large_es-en", [("checkpoint_best.pt", "Health (Fairseq; large)")]),
        #
        # ("biological_fairseq_es-en", [("checkpoint_best.pt", "Biological (Fairseq; small)")]),
        # ("biological_fairseq_large_es-en", [("checkpoint_best.pt", "Biological (Fairseq; large)")]),
        # #
        # ("merged_fairseq_es-en", [("checkpoint_best.pt", "H+B (Fairseq; small)")]),
        # ("merged_fairseq_large_es-en", [("checkpoint_best.pt", "H+B (Fairseq; large)")]),

        # ("health_biological_fairseq_es-en", [("checkpoint_best.pt", "H→B (Fairseq; small)")]),
        # ("health_biological_fairseq_large_es-en", [("checkpoint_best.pt", "H→B (Fairseq; large)")]),


        # Vocabulary domain ***********
        # ("health_fairseq_es-en", [("checkpoint_best.pt", "Health (Voc. domain=Health)")]),
        # ("health_fairseq_vbiological_es-en", [("checkpoint_best.pt", "Health (Voc. domain=Biological)")]),
        # ("health_fairseq_vmerged_es-en", [("checkpoint_best.pt", "Health (Voc. domain=Merged)")]),

        # ("biological_fairseq_vhealth_es-en", [("checkpoint_best.pt", "Biological (Voc. domain=Health)")]),
        # ("biological_fairseq_es-en", [("checkpoint_best.pt", "Biological (Voc. domain=Biological)")]),
        # ("biological_fairseq_vmerged_es-en", [("checkpoint_best.pt", "Biological (Voc. domain=Merged)")]),

        # ("merged_fairseq_vhealth_es-en", [("checkpoint_best.pt", "Merged (Voc. domain=Health)")]),
        # ("merged_fairseq_vbiological_es-en", [("checkpoint_best.pt", "Merged (Voc. domain=Biological)")]),
        # ("merged_fairseq_es-en", [("checkpoint_best.pt", "Merged (Voc. domain=Merged)")]),

        # ("health_fairseq_es-en", [("checkpoint_best.pt", "Health (Voc. domain=H)")]),
        # ("health_biological_fairseq_es-en", [("checkpoint_best.pt", "H→B (Voc. domain=H)")]),
        # # #
        # ("health_fairseq_vbiological_es-en", [("checkpoint_best.pt", "Health\n(Voc. domain=B)")]),
        # ("health_biological_fairseq_vbiological_es-en", [("checkpoint_best.pt", "H→B\n(Voc. domain=B)")]),
        #
        # ("health_fairseq_vmerged_es-en", [("checkpoint_best.pt", "Health\n(Voc. domain=M)")]),
        # ("health_biological_fairseq_vmerged_es-en", [("checkpoint_best.pt", "H→B\n(Voc. domain=M)")]),

        # Custom ***********
        # ("health_es-en", [("transformer_health_best.pt", "Health (Custom)")]),
        # ("biological_es-en", [("transformer_biological_best.pt", "Biological (Custom)")]),
        # ("merged_es-en", [("transformer_merged_best.pt", "H+B (Custom)")]),

        # Interpolation ***********
        # ("health_biological_inter_es-en", [
        #     ("transformer_health_biological_inter_a0.0_best.pt", "H→B (Inter; a=0.0)"),
        #     ("transformer_health_biological_inter_a0.25_best.pt", "H→B (Inter; a=0.25)"),
        #     ("transformer_health_biological_inter_a0.5_best.pt", "H→B (Inter; a=0.50)"),
        #     ("transformer_health_biological_inter_a0.75_best.pt", "H→B (Inter; a=0.75)"),
        #     ]
        #  ),

        # LearningWithoutForgetting ***********
        # ("health_biological_lwf_es-en", [
        #     ("transformer_health_biological_lwf_a0.25_best.pt", "H→B (LwF; a=0.25)"),
        #     ("transformer_health_biological_lwf_a0.5_best.pt", "H→B (LwF; a=0.50)"),
        #     ("transformer_health_biological_lwf_a0.75_best.pt", "H→B (LwF; a=0.75)"),
        # ]),
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
    print(df)
    df.to_csv(os.path.join(summary_path, f"test_data{file_title}.csv"))
    print("Data saved!")

    # Plot metrics
    plot_metrics(df, savepath=summary_path, lang_pair=lang_pair, metric=metric, file_title=file_title)
    print("Done!")
