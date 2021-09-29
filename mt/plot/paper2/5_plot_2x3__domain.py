import math
import os
import random
import time
from pathlib import Path
import json

from mt.plot.data import experiments

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


def plot_bleu_bpe(show_values=True, metric_id="sacrebleu_bleu", metric_name="BLEU", savepath="."):
    sns.set(font_scale=2.5)  # crazy big
    title = f""
    filename = f"2x3_domain__{metric_name}".lower()

    # Build pandas dataframe
    df = pd.DataFrame(data=experiments)

    # Define subplots => px, py = w*dpi, h*dpi  # pixels
    size = (2, 3)  # H, W
    scales = (6, 12)  # H, W
    fig, axes = plt.subplots(size[0], size[1], figsize=(scales[1]*size[1], scales[0]*size[0]), dpi=150)  # (nrows, ncols), (W, H), dpi

    # Image 1
    data = df.loc[df['id'].isin({"4.1", "4.2", "4.3", "4.4"})]
    axes[0, 0].set_title("CommonCrawl")
    g0 = sns.barplot(ax=axes[0, 0], data=data, x="name", y=metric_id, hue="bpe")

    # Image 2
    data = df.loc[df['id'].isin({"3.1", "3.2", "3.3", "3.4"})]
    axes[0, 1].set_title("SciELO (Health domain)")
    g1 = sns.barplot(ax=axes[0, 1], data=data, x="name", y=metric_id, hue="bpe")

    # Image 3
    data = df.loc[df['id'].isin({"3.1", "3.2", "3.3", "3.4"})]
    axes[0, 2].set_title("SciELO (Biological domain)")
    g2 = sns.barplot(ax=axes[0, 2], data=data, x="name", y=metric_id, hue="bpe")

    # Image 4
    data = df.loc[df['id'].isin({"5.1", "5.2", "5.3", "5.4"})]
    axes[1, 0].set_title("NewsCommentary")
    g3 = sns.barplot(ax=axes[1, 0], data=data, x="name", y=metric_id, hue="bpe")

    # Image 5
    data = df.loc[df['id'].isin({"7.1", "7.2"})]
    axes[1, 1].set_title("IWLST2016")
    g5 = sns.barplot(ax=axes[1, 1], data=data, x="name", y=metric_id, hue="bpe")

    # Image 6
    data = df.loc[df['id'].isin({"6.1", "6.2"})]
    axes[1, 2].set_title("Multi30k")
    g4 = sns.barplot(ax=axes[1, 2], data=data, x="name", y=metric_id, hue="bpe")


    for gx in [g0, g1, g2, g3, g4, g5]:
        gx.set(xlabel='', ylabel=metric_name.upper())  #
        # gx.set_xticklabels(rotation=0, horizontalalignment="center")
        gx.set_ylim([0, 50])

        # Add values
        if show_values:
            for c in gx.containers:
                labels = [f"{float(v.get_height()):.1f}" for v in c]
                gx.bar_label(c, labels=labels, label_type='edge')

    # Fix title legend
    legend_loc = "lower right"
    axes[0, 0].legend(title="", loc=legend_loc)
    axes[0, 1].legend(title="", loc=legend_loc)
    axes[0, 2].legend(title="", loc=legend_loc)
    axes[1, 0].legend(title="", loc=legend_loc)
    axes[1, 1].legend(title="", loc=legend_loc)
    axes[1, 2].legend(title="", loc=legend_loc)

    # properties
    plt.ylim([0, 50])
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


if __name__ == "__main__":
    # Plot results (leyend BPE)
    plot_bleu_bpe(savepath=summary_path)

    print("Done!")
