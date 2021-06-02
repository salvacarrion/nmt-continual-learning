import os
import subprocess

import random
import time
from pathlib import Path
import json

from mt import utils
from mt import DATASETS_PATH, DATASET_EVAL_NAME, DATASET_CLEAN_SORTED_NAME, DATASET_TOK_NAME, DATASET_LOGS_NAME, DATASET_CHECKPOINT_NAME

TOK_MODEL = "bpe"
TOK_SIZE = 16000
TOK_FOLDER = f"{TOK_MODEL}.{TOK_SIZE}"
DOMAINS = ["health", "biological", "merged"]


def generate(train_dataset, src, trg, model_name, train_domain):
    # Get all folders in the root path
    test_datasets = [os.path.join(DATASETS_PATH, x) for x in [f"health_{src}-{trg}",
                                                              f"biological_{src}-{trg}",
                                                              f"merged_{src}-{trg}"]]
    for test_dataset in test_datasets:
        test_domain, (test_src, test_trg) = utils.get_dataset_ids(test_dataset)
        print("#############################################")
        print(f"=> TESTING MODEL FROM '{train_domain}' IN DOMAIN '{test_domain}'")

        # Create path
        eval_path = os.path.join(train_dataset, DATASET_EVAL_NAME, model_name, test_domain)
        Path(eval_path).mkdir(parents=True, exist_ok=True)

        # # Preprocess domain datasets with train tokenizers
        # source_dataset = test_dataset
        # vocab_path = train_dataset
        # output_path = eval_path
        # print(f"\t- Preprocessing datasets for: {test_domain}...")
        # subprocess.call(['sh', './scripts/3_preprocess.sh', source_dataset, vocab_path, output_path, TOK_FOLDER, src, trg])

        # Generate them
        eval_path_bin = os.path.join(eval_path, "data-bin")
        model_path = os.path.join(train_dataset, "checkpoints", model_name)
        output_path = eval_path  #os.path.join(eval_path, "generated")
        print(f"\t- Generating translations for: {test_domain}...")
        subprocess.call(['sh', './scripts/5_generate.sh', eval_path_bin, model_path, output_path, src, trg])

        print("")
        print("########################################################################")
        print("########################################################################")
        print("")
    print("")
    print("------------------------------------------------------------------------")
    print("------------------------------------------------------------------------")
    print("")


if __name__ == "__main__":
    # Get all folders in the root path
    # datasets = [os.path.join(DATASETS_PATH, x) for x in ["health_es-en", "biological_es-en", "merged_es-en"]]
    datasets = [(os.path.join(DATASETS_PATH, x), l) for x, l in [
        ("health_fairseq_es-en", ["checkpoint_best.pt"]),
      ]]

    for dataset, models in datasets:
        domain, (src, trg) = utils.get_dataset_ids(dataset)
        fname_base = f"{domain}_{src}-{trg}"


        # Train model
        for model_name in models:
            print(f"Testing model ({fname_base}; {model_name})...")
            generate(dataset, src, trg, model_name=model_name, train_domain=domain)
