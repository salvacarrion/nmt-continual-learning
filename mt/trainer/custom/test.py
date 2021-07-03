import os
import random
import time
from pathlib import Path
import json
import subprocess
import re

import numpy as np
import torch
import torch.nn as nn

import torchtext
from torch.utils.data import DataLoader

from mt import DATASETS_PATH, DATASET_CLEAN_NAME, DATASET_EVAL_NAME, DATASET_TOK_NAME, DATASET_LOGS_NAME, DATASET_CHECKPOINT_NAME
from mt import helpers
from mt import utils
from mt.dataloaders.datasets import TranslationDataset

from mt.trainer.models.transformer.transformer import Transformer
from mt.trainer.custom import base

from torch.utils.data.sampler import SequentialSampler
from torchnlp.samplers import BucketBatchSampler
from mt.dataloaders.max_tokens_batch_sampler import MaxTokensBatchSampler

import sacrebleu
from datasets import load_metric
from torchtext.data.metrics import bleu_score

BATCH_SIZE = 64 #int(32*1.5)
MAX_TOKENS = 2048  #int(4096*1.5)
DEVICE1 = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # torch.device("cpu") #
NUM_WORKERS = 0  # If is not zero, the debugger freezes!!!
TOK_MODEL = "bpe"
TOK_SIZE = 16000
TOK_FOLDER = f"{TOK_MODEL}.{TOK_SIZE}"
LOWERCASE = False
TRUNCATE = True
MAX_LENGTH_TRUNC = 2000
SAMPLER_NAME = "maxtokens"
MAX_LENGTH = 200  # Minimal impact in performance
PRINT_TRANSLATIONS = False
BEAMS = [1]

print(f"Device #1: {DEVICE1}")

###########################################################################
###########################################################################

# Deterministic environment
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def run_experiment(datapath, src, trg, model_name, train_domain):
    start_time = time.time()

    # Load tokenizers
    src_tok, trg_tok = helpers.get_tokenizers(os.path.join(datapath, DATASET_TOK_NAME, TOK_FOLDER), src, trg,
                                              tok_model=TOK_MODEL, lower=LOWERCASE, truncation=TRUNCATE, max_length=MAX_LENGTH_TRUNC)
    # Load dataset
    datapath_clean = DATASET_CLEAN_NAME
    if TOK_MODEL == "bpe":  # Do not preprocess again when using bpe
        src_tok.apply_bpe = False
        trg_tok.apply_bpe = False
        datapath_clean = os.path.join(DATASET_TOK_NAME, TOK_FOLDER)

    # Instantiate model #1
    model = Transformer(d_model=256,
                        enc_layers=3, dec_layers=3,
                        enc_heads=8, dec_heads=8,
                        enc_dff_dim=512, dec_dff_dim=512,
                        enc_dropout=0.1, dec_dropout=0.1,
                        max_src_len=2000, max_trg_len=2000,
                        src_tok=src_tok, trg_tok=trg_tok,
                        static_pos_emb=True).to(DEVICE1)
    print(f'The model has {model.count_parameters():,} trainable parameters')

    # Load weights
    checkpoint_path = os.path.join(datapath, DATASET_CHECKPOINT_NAME, model_name)
    print(f"Loading weights from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path))

    # Other stuff
    criterion = nn.CrossEntropyLoss(ignore_index=trg_tok.word2idx[trg_tok.PAD_WORD])

    # Evaluate on: Health, Biological and Merged domains
    evaluate_hbm(model, criterion, src_tok, trg_tok, train_domain, datapath, datapath_clean, start_time)


def evaluate_hbm(model, criterion, src_tok, trg_tok, train_domain, basepath, datapath_clean, start_time):
    # Get all folders in the root path
    test_datasets = [os.path.join(DATASETS_PATH, TOK_FOLDER, x) for x in [f"health_{src_tok.lang}-{trg_tok.lang}",
                                                              f"biological_{src_tok.lang}-{trg_tok.lang}",
                                                              f"merged_{src_tok.lang}-{trg_tok.lang}"]]
    for test_dataset in test_datasets:
        test_domain, (test_src, test_trg) = utils.get_dataset_ids(test_dataset)
        print("#############################################")
        print(f"=> TESTING MODEL FROM '{train_domain}' IN DOMAIN '{test_domain}'")

        # Get datasets
        test_ds = TranslationDataset(os.path.join(test_dataset, datapath_clean), src_tok, trg_tok, "test")

        # Get dataloaders
        test_loader = base.get_data_loader(SAMPLER_NAME, test_ds, BATCH_SIZE, MAX_TOKENS, NUM_WORKERS, shuffle=False)

        # # Evaluate
        start_time2 = time.time()
        val_loss, val_translations = base.evaluate(model, test_loader, criterion, device=DEVICE1)

        # Log progress
        metrics = base.log_progress(epoch_i=0, start_time=start_time2, tr_loss=None, val_loss=val_loss, tb_writer=None,
                                    translations=val_translations, print_translations=False, prefix=None)

        # Create path
        eval_name = test_domain
        eval_path = os.path.join(basepath, DATASET_EVAL_NAME, model_name, eval_name)
        Path(eval_path).mkdir(parents=True, exist_ok=True)

        # Generate them
        metrics = {"beams": {}}
        for beam in BEAMS:
            print(f"Computing beam width={beam}...")

            # Create output path
            output_path = os.path.join(eval_path, f"beam{beam}")
            Path(output_path).mkdir(parents=True, exist_ok=True)

            print(f"\t- Generating translations for: {test_domain}...")
            # Get translations (using beam search)
            src_dec_all, hyp_dec_all, ref_dec_all = base.get_translations(test_loader, model, device=DEVICE1,
                                                                          max_length=MAX_LENGTH, beam_width=beam)
            # Print translations
            if PRINT_TRANSLATIONS:
                helpers.print_translations(hyp_dec_all, ref_dec_all, src_dec_all, limit=50, randomized=False)

            # Compute scores
            metrics["beams"][f"beam{beam}"] = base.compute_metrics(hyp_dec_all, ref_dec_all, use_ter=False)
            print(f'Translation scores (beam_width={beam}; max_length={MAX_LENGTH})')
            print(f'\t- Sacrebleu (bleu): {metrics[f"beam{beam}"]["sacrebleu_bleu"]:.2f}')
            # print(f'\t- Sacrebleu (ter): {metrics[f"beam{beam}"]["sacrebleu_ter"]:.2f}')
            print(f'\t- Sacrebleu (chrf): {metrics[f"beam{beam}"]["sacrebleu_chrf"]:.2f}')
            print(f'\t- Torchtext (bleu): {metrics[f"beam{beam}"]["torchtext_bleu"]:.2f}')

            # Save translations to file
            with open(os.path.join(output_path, 'src.txt'), 'w') as f:
                f.writelines("%s\n" % s for s in src_dec_all)
            with open(os.path.join(output_path, 'hyp.txt'), 'w') as f:
                f.writelines("%s\n" % s for s in hyp_dec_all)
            with open(os.path.join(output_path, 'ref.txt'), 'w') as f:
                f.writelines("%s\n" % s for s in ref_dec_all)
            print(f"Translations written! => Path: {output_path}")

            # Generate beam metrics
            print(f"\t- Generating translations for: {test_domain}...")
            subprocess.call(['sh', './scripts/6_sacrebleu.sh', eval_path, output_path])
            metrics["beams"].update(get_beam_scores(output_path, beam))

        # Save metrics to file
        with open(os.path.join(eval_path, 'beam_metrics.json'), 'w') as f:
            json.dump(metrics, f)
        print("Metrics saved!")
        print("\t- To get BLEU/CHRF/TER use: 'cat hyp.txt | sacrebleu ref.txt --metrics bleu'")
        print("\t- To get CHRF use: 'chrf -R ref.txt -H hyp.txt'")

        print("************************************************************")
        epoch_hours, epoch_mins, epoch_secs = helpers.epoch_time(start_time, end_time=time.time())
        print(f'Time experiment: {epoch_hours}h {epoch_mins}m {epoch_secs}s')
        print("************************************************************")
        print("Done!")


def get_beam_scores(output_path, beam):
    metrics = {f"beam{beam}": {}}

    # Sacrebleu: BLEU
    with open(os.path.join(output_path, "metrics_bleu.txt"), 'r') as f2:
        score_summary = f2.readlines()[-1]
        print(score_summary)

        # Parse metrics
        pattern = r"BLEU.* = (\d+\.\d+) \d+\.\d+\/"
        score_bleu = re.search(pattern, score_summary).groups()[0]
        score_bleu = float(score_bleu)
        metrics[f"beam{beam}"]['sacrebleu_bleu'] = score_bleu

    # Sacrebleu: CHRF
    with open(os.path.join(output_path, "metrics_chrf.txt"), 'r') as f3:
        score_summary = f3.readlines()[-1]
        print(score_summary)

        # Parse metrics
        pattern = r"chrF2.* = (\d+\.\d+)\s*$"
        score_chrf = re.search(pattern, score_summary).groups()[0]
        score_chrf = float(score_chrf)
        metrics[f"beam{beam}"]['sacrebleu_chrf'] = score_chrf

    # # Sacrebleu: TER
    # with open(os.path.join(output_path, "metrics_ter.txt"), 'r') as f4:
    #     score_summary = f4.readlines()[-1]
    #     print(score_summary)
    #
    #     # Parse metrics
    #     pattern = r"TER.* = (\d+\.\d+)\s*$"
    #     score_ter = re.search(pattern, score_summary).groups()[0]
    #     score_ter = float(score_ter)
    #     metrics[f"beam{beam}"]['sacrebleu_ter'] = score_ter
    print("------------------------------------------------------------------------")
    return metrics


if __name__ == "__main__":
    # Get all folders in the root path
    # datasets = [os.path.join(DATASETS_PATH, x) for x in ["health_es-en", "biological_es-en", "merged_es-en"]]
    datasets = [(os.path.join(DATASETS_PATH, TOK_FOLDER, x), l) for x, l in [
        # ("health_es-en", ["transformer_health_best.pt"]),
        # ("biological_es-en", ["transformer_biological_best.pt"]),
        # ("merged_es-en", ["transformer_merged_best.pt"]),
        ("health_biological_inter_es-en", ["transformer_health_biological_inter_a0.0_best.pt", "transformer_health_biological_inter_a0.25_best.pt", "transformer_health_biological_inter_a0.5_best.pt", "transformer_health_biological_inter_a0.75_best.pt"]),
        ("health_biological_lwf_es-en", ["transformer_health_biological_lwf_a0.25_best.pt", "transformer_health_biological_lwf_a0.5_best.pt", "transformer_health_biological_lwf_a0.75_best.pt"])
    ]]
    # datasets = [os.path.join(DATASETS_PATH, "multi30k_de-en")]
    # datasets = [os.path.join(DATASETS_PATH, "health_es-en")]
    for dataset, models in datasets:
        domain, (src, trg) = utils.get_dataset_ids(dataset)
        fname_base = f"{domain}_{src}-{trg}"

        # Create paths
        Path(os.path.join(dataset, DATASET_LOGS_NAME)).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(dataset, DATASET_CHECKPOINT_NAME)).mkdir(parents=True, exist_ok=True)

        # Train model
        for model_name in models:
            print(f"Testing model ({fname_base}; {model_name})...")
            run_experiment(dataset, src, trg, model_name=model_name, train_domain=domain)
