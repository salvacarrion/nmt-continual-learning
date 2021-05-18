import math
import os
import random
import time
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchtext
from torch.utils.data import DataLoader
from tqdm import tqdm

from mt import DATASETS_PATH, DATASET_CLEAN_NAME, DATASET_CLEAN_SORTED_NAME, DATASET_EVAL_NAME, DATASET_TOK_NAME, DATASET_LOGS_NAME, DATASET_CHECKPOINT_NAME
from mt import helpers
from mt.preprocess import utils
from mt.trainer.datasets import TranslationDataset
from mt.trainer.models.transformer.transformer import Transformer
from mt.trainer.custom import test

from torch.utils.data.sampler import SequentialSampler
from torchnlp.samplers import BucketBatchSampler
from mt.samplers.max_tokens_batch_sampler import MaxTokensBatchSampler


MODEL_NAME = "transformer"


MAX_EPOCHS = 50
LEARNING_RATE = 0.5e-3
BATCH_SIZE = 128 #int(32*1.5)
MAX_TOKENS = 4096  #int(4096*1.5)
WARMUP_UPDATES = 4000
PATIENCE = 10
ACC_GRADIENTS = 1
WEIGHT_DECAY = 0.0001
MULTIGPU = False
DEVICE1 = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # torch.device("cpu") #
DEVICE2 = None  #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0
TOK_MODEL = "bpe"
TOK_SIZE = 16000
TOK_FOLDER = f"{TOK_MODEL}.{TOK_SIZE}"
LOWERCASE = False
SAMPLER_NAME = "maxtokens"
MAX_LENGTH = 50
BEAM_WIDTH = 3

print(f"Device #1: {DEVICE1}")
print(f"Device #2: {DEVICE2}")

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


def run_experiment(datapath, src, trg, model_name, domain=None, smart_batch=False):
    start_time = time.time()

    # Load tokenizers
    src_tok, trg_tok = helpers.get_tokenizers(os.path.join(datapath, DATASET_TOK_NAME, TOK_FOLDER), src, trg, tok_model=TOK_MODEL, lower=LOWERCASE)

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
    criterion = nn.CrossEntropyLoss(ignore_index=trg_tok.word2idx[trg_tok.PAD_WORD])

    # Load weights
    checkpoint_path = os.path.join(datapath, DATASET_CHECKPOINT_NAME, f"transformer_{domain}_best.pt")
    print(f"Loading weights from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path))

    # Get all folders in the root path
    test_datasets = [os.path.join(DATASETS_PATH, x) for x in ["health_es-en", "biological_es-en", "merged_es-en"]]
    for test_dataset in test_datasets:
        test_domain, (test_src, test_trg) = utils.get_dataset_ids(test_dataset)
        print("#############################################")
        print(f"=> TESTING MODEL FROM '{domain}' IN DOMAIN '{test_domain}'")

        # Load clean dataset (untokenized). Each model tokenizer is different
        test_ds = TranslationDataset(os.path.join(test_dataset, DATASET_CLEAN_NAME), src_tok, trg_tok, "test")

        kwargs_test = {}
        if SAMPLER_NAME == "bucket":
            test_sampler = BucketBatchSampler(SequentialSampler(test_ds), batch_size=BATCH_SIZE, drop_last=False,
                                              sort_key=lambda i: len(test_ds.datasets.iloc[i]["src"].split()))
        elif SAMPLER_NAME == "maxtokens":
            test_sampler = MaxTokensBatchSampler(SequentialSampler(test_ds), shuffle=False, batch_size=BATCH_SIZE,
                                                 max_tokens=MAX_TOKENS, drop_last=False,
                                                 sort_key=lambda i: len(test_ds.datasets.iloc[i]["src"].split()))
        else:
            test_sampler = None
            kwargs_test = {"batch_size": BATCH_SIZE, "shuffle": False}

        # Define dataloader
        test_loader = DataLoader(test_ds, num_workers=NUM_WORKERS, collate_fn=lambda x: TranslationDataset.collate_fn(x, MAX_TOKENS), pin_memory=True, batch_sampler=test_sampler, **kwargs_test)

        # Evaluate
        start_time2 = time.time()
        val_loss, translations = test.evaluate(model, test_loader, criterion)

        # Log progress
        metrics = test.log_progress(start_time2, val_loss, translations, print_translations=False)

        # Get bleu
        src_dec_all, hyp_dec_all, ref_dec_all = test.get_translations(test_loader, model, max_length=MAX_LENGTH, beam_width=BEAM_WIDTH)

        # Print translations
        #helpers.print_translations(hyp_dec_all, ref_dec_all, src_dec_all, limit=50)

        # Compute scores
        bleu_score = torchtext.data.metrics.bleu_score([x.split() for x in hyp_dec_all], [[x.split()] for x in ref_dec_all])
        print(f'BLEU score (beam_width={BEAM_WIDTH}; max_length={MAX_LENGTH})= {bleu_score * 100:.2f}')

        # Create path
        eval_name = test_domain
        eval_path = os.path.join(datapath, DATASET_EVAL_NAME, eval_name)
        Path(eval_path).mkdir(parents=True, exist_ok=True)

        # Save translations to file
        with open(os.path.join(eval_path, 'src.txt'), 'w') as f:
            f.writelines("%s\n" % s for s in src_dec_all)
        with open(os.path.join(eval_path, 'hyp.txt'), 'w') as f:
            f.writelines("%s\n" % s for s in hyp_dec_all)
        with open(os.path.join(eval_path, 'ref.txt'), 'w') as f:
            f.writelines("%s\n" % s for s in ref_dec_all)
        print(f"Translations written! => Path: {eval_path}")

        # Save metrics to file
        with open(os.path.join(eval_path, 'metrics.json'), 'w') as f:
            json.dump(metrics, f)
        print("Metrics saved!")
        print("\t- To get BLEU use: 'cat hyp.txt | sacrebleu ref.txt'")

        print("************************************************************")
        epoch_hours, epoch_mins, epoch_secs = helpers.epoch_time(start_time, end_time=time.time())
        print(f'Time experiment: {epoch_hours}h {epoch_mins}m {epoch_secs}s')
        print("************************************************************")
        print("Done!")



if __name__ == "__main__":
    # Get all folders in the root path
    datasets = [os.path.join(DATASETS_PATH, x) for x in ["health_es-en", "biological_es-en", "merged_es-en"]]
    # datasets = [os.path.join(DATASETS_PATH, "multi30k_de-en")]
    # datasets = [os.path.join(DATASETS_PATH, "health_es-en")]
    for dataset in datasets:
        domain, (src, trg) = utils.get_dataset_ids(dataset)
        fname_base = f"{domain}_{src}-{trg}"
        print(f"Testing model ({fname_base})...")

        # Create paths
        Path(os.path.join(dataset, DATASET_LOGS_NAME)).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(dataset, DATASET_CHECKPOINT_NAME)).mkdir(parents=True, exist_ok=True)

        # Train model
        run_experiment(dataset, src, trg, model_name=MODEL_NAME, domain=domain)
