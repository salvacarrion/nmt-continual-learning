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
from mt import utils
from mt.dataloaders.datasets import TranslationDataset

from mt.trainer.models.transformer.transformer import Transformer

from torch.utils.data.sampler import SequentialSampler
from torchnlp.samplers import BucketBatchSampler
from mt.dataloaders.max_tokens_batch_sampler import MaxTokensBatchSampler

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
TRUNCATE = True
MAX_LENGTH_TRUNC = 2000
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
    src_tok, trg_tok = helpers.get_tokenizers(os.path.join(datapath, DATASET_TOK_NAME, TOK_FOLDER), src, trg,
                                              tok_model=TOK_MODEL, lower=LOWERCASE, truncation=TRUNCATE, max_length=MAX_LENGTH_TRUNC)

    # Load dataset
    datapath_clean = DATASET_CLEAN_SORTED_NAME if smart_batch else DATASET_CLEAN_NAME
    if TOK_MODEL == "bpe":  # Do not preprocess again when using bpe
        src_tok.apply_bpe = False
        trg_tok.apply_bpe = False
        datapath_clean = os.path.join(DATASET_TOK_NAME, TOK_FOLDER)
    test_ds = TranslationDataset(os.path.join(datapath, datapath_clean), src_tok, trg_tok, "test")

    kwargs_test = {}
    if SAMPLER_NAME == "bucket":
        test_sampler = BucketBatchSampler(SequentialSampler(test_ds), batch_size=BATCH_SIZE, drop_last=False,
                                         sort_key=lambda i: len(test_ds.datasets.iloc[i]["src"].split()))
    elif SAMPLER_NAME == "maxtokens":
        test_sampler = MaxTokensBatchSampler(SequentialSampler(test_ds), shuffle=False, batch_size=BATCH_SIZE,
                                             max_tokens=MAX_TOKENS, drop_last=False, sort_key=lambda i: len(test_ds.datasets.iloc[i]["src"].split()))
    else:
        test_sampler = None
        kwargs_test = {"batch_size": BATCH_SIZE, "shuffle": False}

    # Define dataloader
    test_loader = DataLoader(test_ds, num_workers=NUM_WORKERS, collate_fn=lambda x: TranslationDataset.collate_fn(x, MAX_TOKENS), pin_memory=True, batch_sampler=test_sampler, **kwargs_test)

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
    checkpoint_path = os.path.join(datapath, DATASET_CHECKPOINT_NAME, "transformer_multi30k_best.pt")
    print(f"Loading weights from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path))

    # Evaluate
    start_time2 = time.time()
    val_loss, translations = evaluate(model, test_loader, criterion)

    # Log progress
    metrics = log_progress(start_time2, val_loss, translations, print_translations=True)

    # Get bleu
    src_dec_all, hyp_dec_all, ref_dec_all = get_translations(test_loader, model, max_length=MAX_LENGTH, beam_width=BEAM_WIDTH)

    # Print translations
    helpers.print_translations(hyp_dec_all, ref_dec_all, src_dec_all, limit=50)

    # Compute scores
    bleu_score = torchtext.data.metrics.bleu_score([x.split() for x in hyp_dec_all], [[x.split()] for x in ref_dec_all])
    print(f'BLEU score (beam_width={BEAM_WIDTH}; max_length={MAX_LENGTH})= {bleu_score * 100:.2f}')

    # Create path
    eval_name = domain
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


def evaluate(model, data_loader, criterion):
    epoch_loss = 0
    src_dec_all, hyp_dec_all, ref_dec_all = [], [], []

    model.eval()
    for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
        try:
            with torch.no_grad():
                # Get batch data
                src, src_mask, trg, trg_mask = [x.to(DEVICE1) for x in batch]

                # Get output
                # output, _ = model(src, trg[:, :-1])
                output, _ = model(src, src_mask, trg[:, :-1], trg_mask[:, :-1])
                _output = output.contiguous().view(-1, output.shape[-1])
                _trg = trg[:, 1:].contiguous().view(-1)

                # Compute loss
                loss = criterion(_output, _trg.long())
                epoch_loss += loss.item()

                # Generate translations (fast)
                hyp_dec_all += model.trg_tok.decode(output.argmax(2), remove_special_tokens=True)
                ref_dec_all += model.trg_tok.decode(trg, remove_special_tokens=True)
                src_dec_all += model.src_tok.decode(src, remove_special_tokens=True)
        except RuntimeError as e:
            print("ERROR BATCH: " + str(i+1))
            print(e)

    return epoch_loss / len(data_loader), (src_dec_all, hyp_dec_all, ref_dec_all)


def log_progress(start_time, val_loss, translations=None, print_translations=True):
    metrics = {
        "val": {
            "loss": val_loss,
            "ppl": math.exp(val_loss),
        },
    }

    # Get additional metrics
    if translations:
        src_dec_all, hyp_dec_all, ref_dec_all = translations
        m_bleu_score = torchtext.data.metrics.bleu_score([x.split(" ") for x in hyp_dec_all], [[x.split(" ")] for x in ref_dec_all])
        metrics["val"]["bleu"] = m_bleu_score*100

        # Print translations
        if print_translations:
            helpers.print_translations(hyp_dec_all, ref_dec_all, src_dec_all, limit=50, randomized=True)

    # Print stuff
    end_time = time.time()
    epoch_hours, epoch_mins, epoch_secs = helpers.epoch_time(start_time, end_time)
    print("------------------------------------------------------------")
    print(f'Evaluate | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\t- Val Loss: {metrics["val"]["loss"]:.3f} | Val PPL: {metrics["val"]["ppl"]:.3f} | Val BLEU: {metrics["val"]["bleu"]:.3f}')
    print("------------------------------------------------------------")

    return metrics


def get_translations(data, model, max_length=50, beam_width=3):
    src_dec_all, hyp_dec_all, ref_dec_all = [], [], []

    for batch in tqdm(data, total=len(data)):
        # Get batch data
        src, src_mask, trg, trg_mask = [x.to(DEVICE1) for x in batch]
        batch_size, src_max_len, trg_max_len = src.shape[0], src.shape[1], trg.shape[1]

        pred_trg, probs, _ = model.translate_batch(src, max_length=max_length, beam_width=beam_width)
        pred_trg = pred_trg[0]

        # cut off <eos> token
        hyp_dec_all += model.trg_tok.decode(pred_trg, remove_special_tokens=True)
        ref_dec_all += model.trg_tok.decode(trg, remove_special_tokens=True)
        src_dec_all += model.src_tok.decode(src, remove_special_tokens=True)

    return src_dec_all, hyp_dec_all, ref_dec_all


if __name__ == "__main__":
    # Get all folders in the root path
    datasets = [os.path.join(DATASETS_PATH, x) for x in ["health_es-en", "biological_es-en", "merged_es-en"]]
    # datasets = [os.path.join(DATASETS_PATH, "multi30k_de-en")]
    # datasets = [os.path.join(DATASETS_PATH, "health_es-en")]
    for dataset in datasets:
        domain, (src, trg) = utils.get_dataset_ids(dataset)
        fname_base = f"{domain}_{src}-{trg}"
        print(f"Training model ({fname_base})...")

        # Create paths
        Path(os.path.join(dataset, DATASET_LOGS_NAME)).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(dataset, DATASET_CHECKPOINT_NAME)).mkdir(parents=True, exist_ok=True)

        # Train model
        run_experiment(dataset, src, trg, model_name=MODEL_NAME, domain=domain)
