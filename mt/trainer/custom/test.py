import os
import numpy as np
import random
import time
import math
from pathlib import Path
from einops import rearrange
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import torchtext
import sacrebleu
from datasets import load_metric

from tqdm import tqdm

from mt.preprocess import utils
from mt import helpers
from mt import DATASETS_PATH, DATASET_LOGS_NAME, DATASET_CHECKPOINT_NAME
from mt.trainer.models.pytransformer.transformer import TransformerModel
from mt.trainer.models.optim import  ScheduledOptim

MODEL_NAME = "transformer"
BPE_FOLDER = "bpe.16000"


MAX_EPOCHS = 50
LEARNING_RATE = 1e-3
BATCH_SIZE = 4#int(32*1.5)
MAX_TOKENS = int(4096*1.5)
WARMUP_UPDATES = 4000
PATIENCE = 10
ACC_GRADIENTS = 8
WEIGHT_DECAY = 0.0001
MULTIGPU = False
DEVICE1 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
DEVICE2 = None  #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

###########################################################################
###########################################################################


def run_experiment(datapath, src, trg, model_name, bpe_folder, domain=None, num_workers=0):
    checkpoint_path = os.path.join(datapath, DATASET_CHECKPOINT_NAME, f"{model_name}_{domain}_best.pt")

    # Load tokenizers
    src_tok, trg_tok = helpers.get_tokenizers(os.path.join(datapath, bpe_folder), src, trg, use_fastbpe=True)  # use_fastbpe != apply_fastbpe

    # Load dataset
    datasets = helpers.load_dataset(os.path.join(datapath, bpe_folder), src, trg, splits=["test"])

    # Prepare data loaders
    test_loader = helpers.build_dataloader(datasets["test"], src_tok, trg_tok, batch_size=BATCH_SIZE, max_tokens=MAX_TOKENS, num_workers=num_workers, shuffle=False)

    # Instantiate model #1
    model1 = TransformerModel(src_tok=src_tok, trg_tok=trg_tok)
    model1.load_state_dict(torch.load(checkpoint_path))
    model1.to(DEVICE1)

    evaluate(model1, test_loader, print_translations=True)

    print("Done!")


def evaluate(model, data_loader, print_translations=True):
    total_loss = 0.0
    all_metrics = []
    start_time = time.time()
    src_dec_all, ref_dec_all, hyp_dec_all = [], [], []

    model.eval()
    for i, batch in enumerate(data_loader):
        with torch.no_grad():
            # Get batch data
            src, src_mask, trg, trg_mask = [x.to(DEVICE1) for x in batch]
            src_vocab1, trg_vocab1 = model.src_tok.get_vocab_size(), model.trg_tok.get_vocab_size()
            batch_size, src_max_len, trg_max_len = src.shape[0], src.shape[1], trg.shape[1]

            src_key_padding_mask = ~src_mask.type(torch.bool).to(DEVICE1)
            tgt_key_padding_mask = ~trg_mask.type(torch.bool).to(DEVICE1)
            memory_key_padding_mask = src_key_padding_mask.clone()

            # Create tgt_inp and tgt_out (which is tgt_inp but shifted by 1)
            tgt_inp, tgt_out = trg[:, :-1], trg[:, 1:]
            #tgt_mask = gen_nopeek_mask(tgt_inp.shape[1]).to(DEVICE1)

            # Get output
            outputs = model.translate_batch(src, src_key_padding_mask, max_length=100, beam_width=1)

            # Get translations
            trg_pred = [x[0][0] for x in outputs]  # Get best
            src_dec, ref_dec, hyp_dec = get_translations(src, trg, trg_pred, model.src_tok, model.trg_tok)
            if print_translations:
                src_dec_all += src_dec
                ref_dec_all += ref_dec
                hyp_dec_all += hyp_dec

    # Print translations
    if print_translations:
        helpers.print_translations(hypothesis=hyp_dec_all, references=ref_dec_all, source=src_dec_all, limit=50)

    return total_loss / len(data_loader), all_metrics


def log_progress(prefix, total_loss, epoch_i, batch_i, n_batches, start_time, tb_writer=None, translations=None):
    elapsed = time.time() - start_time
    sec_per_batch = elapsed / batch_i
    total_minibatches = (epoch_i - 1) * n_batches + batch_i

    # Compute metrics
    metrics = {
        "loss": total_loss / batch_i,
        "ppl": math.exp(total_loss / batch_i),
    }

    if translations:
        src_dec, ref_dec, hyp_dec = translations

        # Compute metrics
        torch_bleu = torchtext.data.metrics.bleu_score([x.split(" ") for x in hyp_dec], [[x.split(" ")] for x in ref_dec])
        metrics["torch_bleu"] = torch_bleu

        # hg_bleu = load_metric("bleu").compute(predictions=[x.split(" ") for x in hyp_dec], references=[[x.split(" ")] for x in ref_dec])
        # metrics["hg_bleu"] = hg_bleu["bleu"]
        #
        # hg_sacrebleu = load_metric("sacrebleu").compute(predictions=hyp_dec, references=[[x] for x in ref_dec])
        # metrics["hg_sacrebleu"] = hg_sacrebleu["score"]

        # metrics["ter"] = 0
        # metrics["chrf"] = 0

    # Print stuff
    str_metrics = "| ".join(["{}: {:.3f}".format(k, v) for k, v in metrics.items()])
    print('[{}] | Epoch: #{:<3} | {:>3}/{:} batches | {:.2f} sec/it (est.: {:.2f} min) || {}'.format(
        prefix.title(), epoch_i, batch_i, n_batches, sec_per_batch, (n_batches-batch_i)*sec_per_batch/60, str_metrics))

    # Tensorboard
    if tb_writer:
        for k, v in metrics.items():
            tb_writer.add_scalar(f'{prefix}_{k.lower()}', v, total_minibatches)
            wandb.log({f'{prefix}_{k.lower()}': v})

    return metrics


def get_translations(src, trg, trg_pred, src_tok, trg_tok):
    # Decode tensors
    src_dec = src_tok.decode(src)
    ref_dec = trg_tok.decode(trg)
    hyp_dec = trg_tok.decode(trg_pred)
    return src_dec, ref_dec, hyp_dec


if __name__ == "__main__":
    # Get all folders in the root path
    #datasets = [os.path.join(DATASETS_PATH, name) for name in os.listdir(DATASETS_PATH) if os.path.isdir(os.path.join(DATASETS_PATH, name))]
    datasets = [os.path.join(DATASETS_PATH, "health_es-en")]
    for dataset in datasets:
        domain, (src, trg) = utils.get_dataset_ids(dataset)
        fname_base = f"{domain}_{src}-{trg}"
        print(f"Training model ({fname_base})...")

        # Create paths
        Path(os.path.join(dataset, DATASET_LOGS_NAME)).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(dataset, DATASET_CHECKPOINT_NAME)).mkdir(parents=True, exist_ok=True)

        # Train model
        run_experiment(dataset, src, trg, model_name=MODEL_NAME, bpe_folder=BPE_FOLDER, domain=domain)
