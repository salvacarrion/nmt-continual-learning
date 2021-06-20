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


TOK_MODEL = "bpe"
TOK_SIZE = 32000
TOK_FOLDER = f"{TOK_MODEL}.{TOK_SIZE}"
VOCAB_STR = str(TOK_SIZE)[:-3] + "k"
BEAM_FOLDER = "beam5"
METRIC = "bleu"


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
            row["label"] = label
            row["train_domain"] = train_domain.title()
            row["test_domain"] = test_domain.title()
            row["lang"] = f"{src}-{trg}"
            row["vocab_size"] = VOCAB_STR
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
    plt.title(f"{metric_name.upper()} scores in different domains | {VOCAB_STR} | {lang_pair}")

    g.set_xticklabels(rotation=0, horizontalalignment="center")
    plt.legend(loc='lower right')
    plt.tight_layout()

    # Save figure
    plt.savefig(os.path.join(savepath, f"{metric_id}_{VOCAB_STR}_scores_{lang_pair}{file_title}.pdf"))
    plt.savefig(os.path.join(savepath, f"{metric_id}_{VOCAB_STR}_scores_{lang_pair}{file_title}.svg"))
    plt.savefig(os.path.join(savepath, f"{metric_id}_{VOCAB_STR}_scores_{lang_pair}{file_title}.png"))
    print("Figures saved!")

    # Show plot
    plt.show()


if __name__ == "__main__":
    metrics = []

    # Get all folders in the root path
    lang_pair = "es-en"
    file_title = "__" + "health_biological_domainM"  #"model_size_health_biological" #"hbm_basic"  #"__"
    metric = ("fairseq_bleu", "bleu")  # (ID, pretty name)
    datasets = [(os.path.join(DATASETS_PATH, TOK_FOLDER, x), l) for x, l in [

        # Basic ***********
        # ("health_fairseq_vhealth_es-en", [("checkpoint_best.pt", "Health\n(small; VD=H)")]),
        # ("biological_fairseq_vbiological_es-en", [("checkpoint_best.pt", "Biological\n(small; VD=B)")]),
        # ("merged_fairseq_vmerged_es-en", [("checkpoint_best.pt", "H+B\n(small; VD=M)")]),

        # Model size (1k only) ***********
        # ("health_fairseq_vhealth_es-en", [("checkpoint_best.pt", "Health\n(small; VD=H)")]),
        # ("health_fairseq_large_vhealth_es-en", [("checkpoint_best.pt", "Health\n(large; VD=H)")]),
        #
        # ("biological_fairseq_vbiological_es-en", [("checkpoint_best.pt", "Biological\n(small; VD=B)")]),
        # ("biological_fairseq_large_vbiological_es-en", [("checkpoint_best.pt", "Biological\n(large; VD=B)")]),
        # #
        # ("merged_fairseq_vmerged_es-en", [("checkpoint_best.pt", "H+B\n(small; VD=M)")]),
        # ("merged_fairseq_large_vmerged_es-en", [("checkpoint_best.pt", "H+B\n(large; VD=M)")]),
        #
        # ("health_biological_fairseq_vhealth_es-en", [("checkpoint_best.pt", "H→B\n(small; VD=H)")]),
        # ("health_biological_fairseq_large_vhealth_es-en", [("checkpoint_best.pt", "H→B\n(large; VD=H)")]),


        # Vocabulary domain ***********
        # ("health_fairseq_vhealth_es-en", [("checkpoint_best.pt", "Health\n(small; VD=H)")]),
        # ("health_fairseq_vbiological_es-en", [("checkpoint_best.pt", "Health\n(small; VD=B)")]),
        # ("health_fairseq_vmerged_es-en", [("checkpoint_best.pt", "Health\n(small; VD=M)")]),
        # #
        # ("biological_fairseq_vhealth_es-en", [("checkpoint_best.pt", "Biological\n(small; VD=H)")]),
        # ("biological_fairseq_vbiological_es-en", [("checkpoint_best.pt", "Biological\n(small; VD=B)")]),
        # ("biological_fairseq_vmerged_es-en", [("checkpoint_best.pt", "Biological\n(small; VD=M)")]),
        #
        # ("merged_fairseq_vhealth_es-en", [("checkpoint_best.pt", "Merged\n(small; VD=H)")]),
        # ("merged_fairseq_vbiological_es-en", [("checkpoint_best.pt", "Merged\n(small; VD=B)")]),
        # ("merged_fairseq_vmerged_es-en", [("checkpoint_best.pt", "Merged\n(small; VD=M)")]),

        # ("health_fairseq_vhealth_es-en", [("checkpoint_best.pt", "Health\n(small; VD=H)")]),
        # ("health_biological_fairseq_vhealth_es-en", [("checkpoint_best.pt", "H→B\n(small; VD=H)")]),

        # ("health_fairseq_vbiological_es-en", [("checkpoint_best.pt", "Health\n(small; VD=B)")]),
        # ("health_biological_fairseq_vbiological_es-en", [("checkpoint_best.pt", "H→B\n(small; VD=B)")]),

        # ("health_fairseq_vmerged_es-en", [("checkpoint_best.pt", "Health\n(small; VD=M)")]),
        # ("health_biological_fairseq_vmerged_es-en", [("checkpoint_best.pt", "H→B\n(small; VD=M)")]),

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
    summary_path = os.path.join(DATASETS_PATH, TOK_FOLDER, DATASET_SUMMARY_NAME, "metrics")
    Path(summary_path).mkdir(parents=True, exist_ok=True)

    # Save data
    df = pd.DataFrame(metrics)
    print(df)
    df.to_csv(os.path.join(summary_path, f"test_data_{VOCAB_STR}_{file_title}.csv"))
    print("Data saved!")

    # Plot metrics
    plot_metrics(df, savepath=summary_path, lang_pair=lang_pair, metric=metric, file_title=file_title)
    print("Done!")
