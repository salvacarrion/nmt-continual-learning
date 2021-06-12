import os
import subprocess

import random
import time
from pathlib import Path
import json
import re

from mt import utils
from mt import DATASETS_PATH, DATASET_EVAL_NAME, DATASET_CLEAN_SORTED_NAME, DATASET_TOK_NAME, DATASET_LOGS_NAME, DATASET_CHECKPOINT_NAME

TOK_MODEL = "bpe"
TOK_SIZE = 16000
TOK_FOLDER = f"{TOK_MODEL}.{TOK_SIZE}"
DOMAINS = ["health", "biological", "merged"]
GET_SCORES = True  # Generate or Get-scores/metrics


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

        # Preprocess domain datasets with train tokenizers
        source_dataset = test_dataset
        vocab_path = train_dataset
        output_path = eval_path
        print(f"\t- Preprocessing datasets for: {test_domain}...")
        subprocess.call(['sh', './scripts/3_preprocess.sh', source_dataset, vocab_path, output_path, TOK_FOLDER, src, trg])

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


def get_scores(train_dataset, src, trg, model_name, train_domain):
    # Get all folders in the root path
    test_datasets = [os.path.join(DATASETS_PATH, x) for x in [f"health_{src}-{trg}",
                                                              f"biological_{src}-{trg}",
                                                              f"merged_{src}-{trg}"]]
    for test_dataset in test_datasets:
        test_domain, (test_src, test_trg) = utils.get_dataset_ids(test_dataset)
        print(f"=> TESTING MODEL FROM '{train_domain}' IN DOMAIN '{test_domain}'")

        # Create path
        eval_path = os.path.join(train_dataset, DATASET_EVAL_NAME, model_name, test_domain)

        # Read file
        with open(os.path.join(eval_path, "generate-test.txt"), 'r') as f:
            score_summary = f.readlines()[-1]
            print(score_summary)

        # Parse metrics
        pattern = r"beam=(\d+): BLEU\d+ = (\d+.\d+)"
        beam_width, score_bleu = re.search(pattern, score_summary).groups()
        beam_width, score_bleu = int(beam_width), float(score_bleu)
        metrics = {f"beam{beam_width}": {'sacrebleu_bleu': score_bleu}}

        # Save metrics to file
        with open(os.path.join(eval_path, 'metrics.json'), 'w') as f:
            json.dump(metrics, f)
        # print("Metrics saved!")
    print("------------------------------------------------------------------------")


if __name__ == "__main__":
    # Get all folders in the root path
    # datasets = [os.path.join(DATASETS_PATH, x) for x in ["health_es-en", "biological_es-en", "merged_es-en"]]
    datasets = [(os.path.join(DATASETS_PATH, x), l) for x, l in [
        # ("health_fairseq_es-en", ["checkpoint_best.pt"]),
        # ("biological_fairseq_es-en", ["checkpoint_best.pt"]),
        # ("merged_fairseq_es-en", ["checkpoint_best.pt"]),
        # ("health_biological_fairseq_es-en", ["checkpoint_best.pt"]),
        #
        # ("health_fairseq_large_es-en", ["checkpoint_best.pt"]),
        # ("biological_fairseq_large_es-en", ["checkpoint_best.pt"]),
        # ("merged_fairseq_large_es-en", ["checkpoint_best.pt"]),
        # ("health_biological_fairseq_large_es-en", ["checkpoint_best.pt"]),

        ("biological_fairseq_vhealth_es-en", ["checkpoint_best.pt"]),
        ("biological_fairseq_vmerged_es-en", ["checkpoint_best.pt"]),
        ("health_fairseq_vbiological_es-en", ["checkpoint_best.pt"]),
        ("health_fairseq_vmerged_es-en", ["checkpoint_best.pt"]),
        ("merged_fairseq_vbiological_es-en", ["checkpoint_best.pt"]),
        ("merged_fairseq_vhealth_es-en", ["checkpoint_best.pt"]),
      ]]

    for dataset, models in datasets:
        domain, (src, trg) = utils.get_dataset_ids(dataset)
        fname_base = f"{domain}_{src}-{trg}"

        # Train model
        for model_name in models:
            print(f"Testing model ({fname_base}; {model_name})...")
            func = get_scores if GET_SCORES else generate
            func(dataset, src, trg, model_name=model_name, train_domain=domain)
