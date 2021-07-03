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
TOK_SIZE = 2000
TOK_FOLDER = f"{TOK_MODEL}.{TOK_SIZE}"
DOMAINS = ["health", "biological", "merged"]
BEAMS = [5]


def generate(train_dataset, src, trg, model_name, train_domain):
    # Get all folders in the root path
    test_datasets = [os.path.join(DATASETS_PATH, TOK_FOLDER, x) for x in [f"health_fairseq_vhealth_{src}-{trg}",
                                                                          f"biological_fairseq_vbiological_{src}-{trg}",
                                                                          f"merged_fairseq_vmerged_{src}-{trg}"]]
    for test_dataset in test_datasets:
        test_domain, (test_src, test_trg) = utils.get_dataset_ids(test_dataset)
        test_domain = test_domain.replace("_fairseq", "").replace("_vhealth", "").replace("_vbiological", "").replace("_vmerged", "")
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
        for beam in BEAMS:
            eval_path_bin = os.path.join(eval_path, "data-bin")
            model_path = os.path.join(train_dataset, "checkpoints", model_name)

            # Create output path
            output_path = os.path.join(eval_path, f"beam{beam}")
            Path(output_path).mkdir(parents=True, exist_ok=True)

            print(f"\t- Generating translations for: {test_domain}...")
            subprocess.call(['sh', './scripts/5_generate.sh', eval_path_bin, model_path, output_path, src, trg, str(beam)])

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
    test_datasets = [os.path.join(DATASETS_PATH, TOK_FOLDER, x) for x in [f"health_fairseq_vhealth_{src}-{trg}",
                                                                          f"biological_fairseq_vbiological_{src}-{trg}",
                                                                          f"merged_fairseq_vmerged_{src}-{trg}"]]
    for test_dataset in test_datasets:
        test_domain, (test_src, test_trg) = utils.get_dataset_ids(test_dataset)
        test_domain = test_domain.replace("_fairseq", "").replace("_vhealth", "").replace("_vbiological", "").replace("_vmerged", "")
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


def get_beam_scores(train_dataset, src, trg):
    domain, (src, trg) = utils.get_dataset_ids(dataset)
    fname_base = f"{domain}_{src}-{trg}"

    # Get all folders in the root path
    test_datasets = [os.path.join(DATASETS_PATH, TOK_FOLDER, x) for x in [f"health_fairseq_vhealth_{src}-{trg}",
                                                                          f"biological_fairseq_vbiological_{src}-{trg}",
                                                                          f"merged_fairseq_vmerged_{src}-{trg}"]]
    metrics = {"beams": {}}

    for test_dataset in test_datasets:
        test_domain, (test_src, test_trg) = utils.get_dataset_ids(test_dataset)
        test_domain = test_domain.replace("_fairseq", "").replace("_vhealth", "").replace("_vbiological", "").replace("_vmerged", "")
        print(f"=> TESTING MODEL FROM '{fname_base}' IN DOMAIN '{test_domain}'")

        # Create path
        eval_path = os.path.join(train_dataset, DATASET_EVAL_NAME, model_name, test_domain)

        # Generate them
        for beam in BEAMS:
            metrics["beams"][f"beam{beam}"] = {}

            # Set output path
            output_path = os.path.join(eval_path, f"beam{beam}")

            # Read fairseq-generate output
            with open(os.path.join(output_path, "generate-test.txt"), 'r') as f:
                score_summary = f.readlines()[-1]
                print(score_summary)

                # Parse metrics
                pattern = r"beam=(\d+): BLEU\d+ = (\d+.\d+)"
                beam_width, score_bleu = re.search(pattern, score_summary).groups()
                beam_width, score_bleu = int(beam_width), float(score_bleu)
                metrics["beams"][f"beam{beam}"]['fairseq_bleu'] = score_bleu

            # Sacrebleu: BLEU
            with open(os.path.join(output_path, "metrics_bleu.txt"), 'r') as f2:
                score_summary = f2.readlines()[-1]
                print(score_summary)

                # Parse metrics
                pattern = r"BLEU.* = (\d+\.\d+) \d+\.\d+\/"
                score_bleu = re.search(pattern, score_summary).groups()[0]
                score_bleu = float(score_bleu)
                metrics["beams"][f"beam{beam}"]['sacrebleu_bleu'] = score_bleu

            # Sacrebleu: CHRF
            with open(os.path.join(output_path, "metrics_chrf.txt"), 'r') as f3:
                score_summary = f3.readlines()[-1]
                print(score_summary)

                # Parse metrics
                pattern = r"chrF2.* = (\d+\.\d+)\s*$"
                score_chrf = re.search(pattern, score_summary).groups()[0]
                score_chrf = float(score_chrf)
                metrics["beams"][f"beam{beam}"]['sacrebleu_chrf'] = score_chrf

            # # Sacrebleu: TER
            # with open(os.path.join(output_path, "metrics_ter.txt"), 'r') as f4:
            #     score_summary = f4.readlines()[-1]
            #     print(score_summary)
            #
            #     # Parse metrics
            #     pattern = r"TER.* = (\d+\.\d+)\s*$"
            #     score_ter = re.search(pattern, score_summary).groups()[0]
            #     score_ter = float(score_ter)
            #     metrics["beams"][f"beam{beam}"]['sacrebleu_ter'] = score_ter

        # Save metrics to file
        with open(os.path.join(eval_path, 'beam_metrics.json'), 'w') as f:
            json.dump(metrics, f)
        print("Metrics saved!")
    print("------------------------------------------------------------------------")


if __name__ == "__main__":
    # Get all folders in the root path
    # datasets = [os.path.join(DATASETS_PATH, x) for x in ["health_es-en", "biological_es-en", "merged_es-en"]]
    datasets = [(os.path.join(DATASETS_PATH, TOK_FOLDER, x), l) for x, l in [
        # ("health_fairseq_es-en", ["checkpoint_best.pt"]),
        # ("biological_fairseq_es-en", ["checkpoint_best.pt"]),
        # ("merged_fairseq_es-en", ["checkpoint_best.pt"]),
        # ("health_biological_fairseq_es-en", ["checkpoint_best.pt"]),
        #
        # ("biological_fairseq_vhealth_es-en", ["checkpoint_best.pt"]),
        # ("biological_fairseq_vmerged_es-en", ["checkpoint_best.pt"]),
        # ("health_fairseq_vbiological_es-en", ["checkpoint_best.pt"]),
        # ("health_fairseq_vmerged_es-en", ["checkpoint_best.pt"]),
        # ("merged_fairseq_vbiological_es-en", ["checkpoint_best.pt"]),
        # ("merged_fairseq_vhealth_es-en", ["checkpoint_best.pt"]),
        #
        # ("health_biological_fairseq_vbiological_es-en", ["checkpoint_best.pt"]),
        # ("health_biological_fairseq_vmerged_es-en", ["checkpoint_best.pt"]),

        # Fairseq (small): Health
        ("health_fairseq_vhealth_es-en", ["checkpoint_best.pt"]),
        ("health_fairseq_vbiological_es-en", ["checkpoint_best.pt"]),
        ("health_fairseq_vmerged_es-en", ["checkpoint_best.pt"]),
        #
        # # Fairseq (small): Biological
        ("biological_fairseq_vhealth_es-en", ["checkpoint_best.pt"]),
        ("biological_fairseq_vbiological_es-en", ["checkpoint_best.pt"]),
        ("biological_fairseq_vmerged_es-en", ["checkpoint_best.pt"]),
        #
        # # Fairseq (small): Merged
        ("merged_fairseq_vhealth_es-en", ["checkpoint_best.pt"]),
        ("merged_fairseq_vbiological_es-en", ["checkpoint_best.pt"]),
        ("merged_fairseq_vmerged_es-en", ["checkpoint_best.pt"]),
        #
        # # Fairseq (small): H->B
        ("health_biological_fairseq_vhealth_es-en", ["checkpoint_best.pt"]),
        ("health_biological_fairseq_vbiological_es-en", ["checkpoint_best.pt"]),
        ("health_biological_fairseq_vmerged_es-en", ["checkpoint_best.pt"]),

        # Fairseq (large): All
        # ("health_fairseq_large_vhealth_es-en", ["checkpoint_best.pt"]),
        # ("biological_fairseq_large_vbiological_es-en", ["checkpoint_best.pt"]),
        # ("merged_fairseq_large_vmerged_es-en", ["checkpoint_best.pt"]),
        # ("health_biological_fairseq_large_vhealth_es-en", ["checkpoint_best.pt"]),
    ]]

    for dataset, models in datasets:
        domain, (src, trg) = utils.get_dataset_ids(dataset)
        fname_base = f"{domain}_{src}-{trg}"

        # Train model
        for model_name in models:
            print(f"Testing model ({fname_base}; {model_name})...")

            generate(dataset, src, trg, model_name=model_name, train_domain=domain)
            # get_scores(dataset, src, trg, model_name=model_name, train_domain=domain)  # Old
            get_beam_scores(dataset, src, trg)
