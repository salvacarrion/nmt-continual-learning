import os
import subprocess
import os

from mt import utils
from mt import DATASETS_PATH, DATASET_CLEAN_NAME, DATASET_CLEAN_SORTED_NAME, DATASET_TOK_NAME, DATASET_LOGS_NAME, DATASET_CHECKPOINT_NAME


WANDB_PROJECT = "nmt2"  # Run "wandb login" in the terminal

TOK_MODEL = "bpe"  # wt
TOK_SIZE = 64
TOK_FOLDER = f"{TOK_MODEL}.{TOK_SIZE}"


def train(datapath):
    # script = "4_train_transformer_small.sh"
    # script = "4_train_transformer_large.sh"
    script = "4_train_cnn.sh"
    # script = "4_train_lstm.sh"
    subprocess.call(['sh', f'./scripts/{script}', datapath, WANDB_PROJECT])


if __name__ == "__main__":
    # Get all folders in the root path
    datasets = [os.path.join(DATASETS_PATH, TOK_FOLDER, x) for x in [
        # "europarl_fairseq_large_es-en",  #

        # "europarl_fairseq_conv_es-en",  #
        "europarl_fairseq_50k_conv_es-en",

        # "europarl_fairseq_lstm_es-en", #
        # "europarl_fairseq_50k_lstm_es-en",

        # "europarl_fairseq_transxs_es-en",
        # "europarl_fairseq_50k_transxs_es-en",

        # "europarl_fairseq_50k_de-en",
        # "europarl_fairseq_50k_es-en",
        # "europarl_fairseq_50k_cs-en",
        # "europarl_fairseq_cs-en",
        # "europarl_fairseq_100k_cs-en",
        # "europarl_fairseq_de-en",
        # "europarl_fairseq_100k_de-en",
        # "europarl_fairseq_de-en",
        # "europarl_fairseq_fr-en",
        # "europarl_fairseq_cs-en",
        # "commoncrawl_es-en",
        # "commoncrawl_100k_es-en",
        # "newscommentaryv14_35k_es-en",
        # "newscommentaryv14_es-en",
        # "europarl_fairseq_100k_es-en",

        # "health_fairseq_vhealth_unconstrained_es-en",
        # "iwlst2016_de-en",
        #"multi30k_de-en",

        # "health_fairseq_vhealth_es-en",
        # "health_fairseq_vbiological_es-en",
        # "health_fairseq_vmerged_es-en",
        #
        # "biological_fairseq_vhealth_es-en",
        # "biological_fairseq_vbiological_es-en",
        # "biological_fairseq_vmerged_es-en",
        #
        # "merged_fairseq_vhealth_es-en",
        # "merged_fairseq_vbiological_es-en",
        # "merged_fairseq_vmerged_es-en",

        # "health_biological_fairseq_vhealth_es-en",
        # "health_biological_fairseq_vbiological_es-en",
        # "health_biological_fairseq_vmerged_es-en",

        # "health_fairseq_large_vhealth_es-en",
        # "biological_fairseq_large_vbiological_es-en",
        # "merged_fairseq_large_vmerged_es-en",
        # "health_biological_fairseq_large_vhealth_es-en",
    ]]
    for dataset in datasets:
        domain, (src, trg) = utils.get_dataset_ids(dataset)
        fname_base = f"{domain}_{src}-{trg}"
        print(f"Train fairseq model ({fname_base})...")

        # Train model
        train(dataset)
