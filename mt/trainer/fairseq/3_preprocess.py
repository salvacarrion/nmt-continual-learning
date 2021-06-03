import os
import subprocess
import os

from mt import utils
from mt import DATASETS_PATH, DATASET_CLEAN_NAME, DATASET_CLEAN_SORTED_NAME, DATASET_TOK_NAME, DATASET_LOGS_NAME, DATASET_CHECKPOINT_NAME

TOK_MODEL = "bpe"
TOK_SIZE = 16000
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
    datasets = [os.path.join(DATASETS_PATH, x) for x in ["health_biological_fairseq_es-en"]]
    for dataset in datasets:
        domain, (src, trg) = utils.get_dataset_ids(dataset)
        fname_base = f"{domain}_{src}-{trg}"
        print(f"Preprocess data for fairseq ({fname_base})...")

        # Preprocess data
        preprocess(dataset, src, trg)
