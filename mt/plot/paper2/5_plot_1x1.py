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
    filename = f"europarl-50k_cs-en__{metric_name}".lower()

    # Build pandas dataframe
    df = pd.DataFrame(data=experiments)

    # Define subplots => px, py = w*dpi, h*dpi  # pixels
    size = (1, 1)  # W, H
    scales = (12, 8)
    fig, axes = plt.subplots(1, figsize=(scales[0]*size[0], scales[1]*size[1]), dpi=150)  # (nrows, ncols), (W, H), dpi

    # Image 00
    data = df.loc[df['id'].isin({"0.5", "0.6"})]
    axes.set_title("Europarl-50K (cs-en)")
    g0 = sns.barplot(ax=axes, data=data, x="name", y=metric_id, hue="bpe")
    #
    # data = df.loc[df['id'].isin({"1.3", "1.4"})]
    # axes[0, 1].set_title("Europarl-100k (es-en)")
    # g1 = sns.barplot(ax=axes[0, 1], data=data, x="name", y=metric_id, hue="bpe")
    #
    # data = df.loc[df['id'].isin({"1.5", "1.6"})]
    # axes[0, 2].set_title("Europarl-50k (es-en)")
    # g2 = sns.barplot(ax=axes[0, 2], data=data, x="name", y=metric_id, hue="bpe")
    #
    # # Image 2
    # data = df.loc[df['id'].isin({"2.1", "2.2"})]
    # axes[1, 0].set_title("Europarl-2M (de-en)")
    # g3 = sns.barplot(ax=axes[1, 0], data=data, x="name", y=metric_id, hue="bpe")
    #
    # data = df.loc[df['id'].isin({"2.3", "2.4"})]
    # axes[1, 1].set_title("Europarl-100k (de-en)")
    # g4 = sns.barplot(ax=axes[1, 1], data=data, x="name", y=metric_id, hue="bpe")
    #
    # data = df.loc[df['id'].isin({"2.5", "2.6"})]
    # axes[1, 2].set_title("Europarl-50k (de-en)")
    # g5 = sns.barplot(ax=axes[1, 2], data=data, x="name", y=metric_id, hue="bpe")
    #
    # # Image 3
    # data = df.loc[df['id'].isin({"0.1", "0.2"})]
    # axes[2, 0].set_title("Europarl-630k (cs-en)")
    # g6 = sns.barplot(ax=axes[2, 0], data=data, x="name", y=metric_id, hue="bpe")
    #
    # # Image 2
    # data = df.loc[df['id'].isin({"0.3", "0.4"})]
    # axes[2, 1].set_title("Europarl-100k (cs-en)")
    # g7 = sns.barplot(ax=axes[2, 1], data=data, x="name", y=metric_id, hue="bpe")
    #
    # # Image 3
    # data = df.loc[df['id'].isin({"0.5", "0.6"})]
    # axes[2, 2].set_title("Europarl-50k (cs-en)")
    # g8 = sns.barplot(ax=axes[2, 2], data=data, x="name", y=metric_id, hue="bpe")

    for gx in [g0]:
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
    axes.legend(title="", loc=legend_loc)
    # axes[0, 1].legend(title="", loc=legend_loc)
    # axes[0, 2].legend(title="", loc=legend_loc)
    # axes[1, 0].legend(title="", loc=legend_loc)
    # axes[1, 1].legend(title="", loc=legend_loc)
    # axes[1, 2].legend(title="", loc=legend_loc)
    # axes[2, 0].legend(title="", loc=legend_loc)
    # axes[2, 1].legend(title="", loc=legend_loc)
    # axes[2, 2].legend(title="", loc=legend_loc)

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
