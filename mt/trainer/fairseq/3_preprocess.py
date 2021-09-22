import os
import subprocess
import os

from mt import utils
from mt import DATASETS_PATH, DATASET_BPE_NAME

TOK_MODEL = "bpe"  # wt
TOK_SIZE = 64
TOK_FOLDER = f"{TOK_MODEL}.{TOK_SIZE}"


def preprocess(train_dataset, src, trg):
    # Preprocess domain datasets with train tokenizers
    source_dataset = train_dataset
    vocab_path = train_dataset
    output_path = train_dataset
    subprocess.call(['sh', './scripts/3_preprocess.sh', source_dataset, vocab_path, output_path, TOK_FOLDER, src, trg])


if __name__ == "__main__":
    # Get all folders in the root path
    # datasets = [os.path.join(DATASETS_PATH, x) for x in ["health_fairseq_es-en", "biological_fairseq_es-en", "merged_fairseq_es-en"]]
    # datasets = [os.path.join(DATASETS_PATH, x) for x in ["health_biological_fairseq_es-en"]]
    datasets = [os.path.join(DATASETS_PATH, TOK_FOLDER, x) for x in [
        "europarl_fairseq_50k_de-en",
        # "europarl_fairseq_50k_es-en",
        # "europarl_fairseq_50k_cs-en",
        # "europarl_fairseq_cs-en",
        # "europarl_fairseq_100k_cs-en",
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
        # "multi30k_de-en",

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
    ]]
    for dataset in datasets:
        domain, (src, trg) = utils.get_dataset_ids(dataset)
        fname_base = f"{domain}_{src}-{trg}"
        print(f"Preprocess data for fairseq ({fname_base})...")

        # Preprocess data
        preprocess(dataset, src, trg)
