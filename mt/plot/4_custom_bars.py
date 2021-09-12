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

# Create folder
summary_path = os.path.join(DATASETS_PATH, "custom_plots")
Path(summary_path).mkdir(parents=True, exist_ok=True)
rows_beam5 = [
    # {"name": "Europarl\n(2M; es-en)", "bpe": "32k vocab", "lang-pair": "es-en", "sacrebleu_bleu": 44.5, "sacrebleu_chrf": 0.67, "sentences": 1948939, "tokens": 55105930, "tokens/sentences": 28.275},
    # {"name": "Europarl\n(2M; es-en)", "bpe": "Quasi Character-level", "lang-pair": "es-en", "sacrebleu_bleu": 39.6, "sacrebleu_chrf": 0.64, "sentences": 1948939, "tokens": 168327841, "tokens/sentences": 86.369},
    # {"name": "Europarl\n(100K; es-en)", "bpe": "32k vocab", "lang-pair": "es-en", "sacrebleu_bleu": 33.0, "sacrebleu_chrf": 0.58, "sentences": 100000, "tokens": 2818061, "tokens/sentences": 28.181},
    # {"name": "Europarl\n(100K; es-en)", "bpe": "Quasi Character-level", "lang-pair": "es-en", "sacrebleu_bleu": 35.2, "sacrebleu_chrf": 0.6, "sentences": 100000, "tokens": 8621638, "tokens/sentences": 86.216},
    #
    # {"name": "Europarl\n(2M; de-en)", "bpe": "32k vocab", "lang-pair": "de-en", "sacrebleu_bleu": 36.6, "sacrebleu_chrf": 0.61, "sentences": 1948939, "tokens": 55105930, "tokens/sentences": 28.275},
    # {"name": "Europarl\n(2M; de-en)", "bpe": "Quasi Character-level", "lang-pair": "de-en", "sacrebleu_bleu": 30.7, "sacrebleu_chrf": 0.57, "sentences": 1948939, "tokens": 168327841, "tokens/sentences": 86.369},
    # {"name": "Europarl\n(100K; de-en)", "bpe": "32k vocab", "lang-pair": "de-en", "sacrebleu_bleu": 24.0, "sacrebleu_chrf": 0.5, "sentences": 100000, "tokens": 2818061, "tokens/sentences": 28.181},
    # {"name": "Europarl\n(100K; de-en)", "bpe": "Quasi Character-level", "lang-pair": "de-en", "sacrebleu_bleu": 26.4, "sacrebleu_chrf": 0.53, "sentences": 100000, "tokens": 8621638, "tokens/sentences": 86.216},
    #
    #
    # {"name": "Biomedical\n(570K; es-en)", "bpe": "32k vocab", "lang-pair": "es-en", "sacrebleu_bleu": 35.6, "sacrebleu_chrf": 0.63, "sentences": 575521, "tokens": 16596128, "tokens/sentences": 28.837},
    # {"name": "Biomedical\n(570K; es-en)", "bpe": "Quasi Character-level", "lang-pair": "es-en", "sacrebleu_bleu": 35.2, "sacrebleu_chrf": 0.63, "sentences": 575521, "tokens": 53033532, "tokens/sentences": 92.149},
    # {"name": "Biomedical\n(120K; es-en)", "bpe": "32k vocab", "lang-pair": "es-en", "sacrebleu_bleu": 28.7, "sacrebleu_chrf": 0.56, "sentences": 118636, "tokens": 3415430, "tokens/sentences": 28.789},
    # {"name": "Biomedical\n(120K; es-en)", "bpe": "Quasi Character-level","lang-pair": "es-en",  "sacrebleu_bleu": 33.3, "sacrebleu_chrf": 0.61, "sentences": 118636, "tokens": 10932149, "tokens/sentences": 92.149},
    #
    # {"name": "CommonCrawl\n(180K; es-en)", "bpe": "32k vocab", "lang-pair": "es-en", "sacrebleu_bleu": 30.7, "sacrebleu_chrf": 0.52, "sentences": 1815333, "tokens": 50918607, "tokens/sentences": 28.049},
    # {"name": "CommonCrawl\n(180K; es-en)", "bpe": "Quasi Character-level", "lang-pair": "es-en", "sacrebleu_bleu": 26.5, "sacrebleu_chrf": 0.49, "sentences": 1815247, "tokens": 147347224, "tokens/sentences": 81.172},
    # {"name": "CommonCrawl\n(100K; es-en)", "bpe": "32k vocab", "lang-pair": "es-en", "sacrebleu_bleu": 15.6, "sacrebleu_chrf": 0.37, "sentences": 100000, "tokens": 2788613, "tokens/sentences": 27.886},
    # {"name": "CommonCrawl\n(100K; es-en)", "bpe": "Quasi Character-level", "lang-pair": "es-en", "sacrebleu_bleu": 22.6, "sacrebleu_chrf": 0.46, "sentences": 99994, "tokens": 8095831, "tokens/sentences": 80.963},
    #
    # {"name": "NewsCommentary\n(360K; es-en)", "bpe": "32k vocab", "lang-pair": "es-en", "sacrebleu_bleu": 45.4, "sacrebleu_chrf": 0.67, "sentences": 357280, "tokens": 9525425, "tokens/sentences": 26.661},
    # {"name": "NewsCommentary\n(360K; es-en)", "bpe": "Quasi Character-level", "lang-pair": "es-en", "sacrebleu_bleu": 42.6, "sacrebleu_chrf": 0.66, "sentences": 357280, "tokens": 30561745, "tokens/sentences": 85.540},
    # {"name": "NewsCommentary\n(35K; es-en)", "bpe": "32k vocab", "lang-pair": "es-en", "sacrebleu_bleu": 21.5, "sacrebleu_chrf": 0.48, "sentences": 35000, "tokens": 928552, "tokens/sentences": 26.530},
    # {"name": "NewsCommentary\n(35K; es-en)", "bpe": "Quasi Character-level", "lang-pair": "es-en", "sacrebleu_bleu": 30.1, "sacrebleu_chrf": 0.57, "sentences": 35000, "tokens": 3000210, "tokens/sentences": 85.720},
    #
    # {"name": "Multi30k\n(30K; de-en)", "bpe": "32k vocab", "lang-pair": "de-en", "sacrebleu_bleu": 32.9, "sacrebleu_chrf": 0.55, "sentences": 29000, "tokens": 380835, "tokens/sentences": 13.132},
    # {"name": "Multi30k\n(30K; de-en)", "bpe": "Quasi Character-level", "lang-pair": "de-en", "sacrebleu_bleu": 34.3, "sacrebleu_chrf": 0.54, "sentences": 29000, "tokens": 991180, "tokens/sentences": 34.179},
    #
    {"name": "IWLST2016\n(200K; de-en)", "bpe": "32k vocab", "lang-pair": "de-en", "sacrebleu_bleu": 28.1, "sacrebleu_chrf": 0.49, "sentences": 196869, "tokens": 4072858, "tokens/sentences": 20.688},
    {"name": "IWLST2016\n(200K; de-en)", "bpe": "Quasi Character-level", "lang-pair": "de-en", "sacrebleu_bleu": 35.2, "sacrebleu_chrf": 0.63, "sentences": 196869, "tokens": 11020739, "tokens/sentences": 55.980},

]  # trH_tsH_vH_


def plot_bleu_bpe(show_values=True, metric_id="sacrebleu_bleu", metric_name="BLEU", savepath="."):
    dataset_name = "IWLST2016"
    title = f"Effects of the vocabulary and training size ({dataset_name})"
    filename = f"{metric_name}_{dataset_name}_de-en__bpe_comparison".lower()

    # Build pandas dataframe
    df = pd.DataFrame(data=rows_beam5)

    # Draw a nested barplot by species and sex
    g = sns.catplot(data=df, x="name", y=metric_id, kind="bar", hue="bpe", legend=False)
    g.fig.set_size_inches(12, 8)

    # Add values
    if show_values:
        ax = g.facet_axis(0, 0)
        for c in ax.containers:
            labels = [f"{float(v.get_height()):.1f}" for v in c]
            ax.bar_label(c, labels=labels, label_type='edge', fontsize=8)

    # properties
    g.set(xlabel='Datasets', ylabel=metric_name.upper())
    plt.title(title)
    plt.ylim([0, 50])

    g.set_xticklabels(rotation=0, horizontalalignment="center")
    plt.legend(loc='upper right')
    plt.tight_layout()

    # Save figure
    plt.savefig(os.path.join(savepath, f"{filename}.pdf"))
    plt.savefig(os.path.join(savepath, f"{filename}.svg"))
    plt.savefig(os.path.join(savepath, f"{filename}.png"))
    print("Figures saved!")

    # Show plot
    plt.show()


def view_tokens():

    datasets = [
        "europarl_fairseq_es-en",
        "europarl_fairseq_100k_es-en",
        "europarl_fairseq_de-en",
        "europarl_fairseq_100k_de-en",
        "health_fairseq_vhealth_unconstrained_es-en",
        "health_fairseq_vhealth_es-en",
        "commoncrawl_es-en",
        "commoncrawl_100k_es-en",
        "newscommentaryv14_es-en",
        "newscommentaryv14_35k_es-en",
        "multi30k_de-en",
        "iwlst2016_de-en",
    ]
    bpes = ["bpe.32000", "bpe.64"]
    base_path = "/home/scarrion/datasets/scielo/constrained/datasets"
    for dataset in datasets:
        print(f"{dataset}: ##############")
        for bpe_i in bpes:

            # Exceptions
            if dataset == "multi30k_de-en" and bpe_i == "bpe.32000":
                bpe_i = "bpe.16000"

            # Get sentences
            with open(os.path.join(base_path, bpe_i, dataset, "tok", bpe_i, "train.en"), "r") as f:
                file = f.readlines()
                tokens = [len(line.strip().split()) for line in file]
                print(f"\t- BPE: {bpe_i} -----------")
                print(f"\t\t- sentences: {len(file)}")
                print(f"\t\t- tokens: {sum(tokens)}")
                print("\t\t- tokens/sentences: {:.3f}".format(sum(tokens)/len(file)))

            # Get scores
            try:
                with open(os.path.join(base_path, bpe_i, dataset, "eval", "checkpoint_best.pt", dataset[:-6], "beam_metrics.json"), "r") as f:
                    print(f"\t\t- Scores: {f.readlines()}")
            except Exception as e:
                print(f"\t\t- Scores error: {e}")

            # Break line
            print("")


if __name__ == "__main__":
    # Plot results (leyend BPE)
    plot_bleu_bpe(savepath=summary_path)

    # View tokens
    # view_tokens()

    print("Done!")
