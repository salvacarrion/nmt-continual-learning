import math
import os
import random
import time
from pathlib import Path
import json

from plot.data import experiments

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
    sns.set(font_scale=1.5)  # crazy big
    title = f"Consistency on other domains"
    filename = f"consistency_de-en_euro__{metric_name}".lower()

    # Build pandas dataframe
    df = pd.DataFrame(data=experiments)

    # Define subplots => px, py = w*dpi, h*dpi  # pixels
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=150)  # (nrows, ncols), (W, H), dpi

    # Image 0
    data = df.loc[df['id'].isin({"2.1", "2.2"})]
    axes[0].set_title("Europarl-Large DE-EN")
    g1 = sns.barplot(ax=axes[0], data=data, x="name", y=metric_id, hue="bpe")

    # Image 1
    data = df.loc[df['id'].isin({"2.3", "2.4"})]
    axes[1].set_title("Europarl-Small DE-EN")
    g2 = sns.barplot(ax=axes[1], data=data, x="name", y=metric_id, hue="bpe")

    for gx in [g1, g2]:
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
    axes[0].legend(title="", loc=legend_loc)
    axes[1].legend(title="", loc=legend_loc)

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
