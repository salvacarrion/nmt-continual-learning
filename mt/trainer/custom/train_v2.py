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
from torch.utils.data import DataLoader

import torchtext
from torchtext.legacy.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator

import torchtext
import sacrebleu
from datasets import load_metric
from torchtext.data.metrics import bleu_score

from tqdm import tqdm
from collections import Counter

from mt.preprocess import utils
from mt import helpers
from mt.trainer.datasets import TranslationDataset
from mt import DATASETS_PATH, DATASET_CLEAN_NAME, DATASET_TOK_NAME, DATASET_LOGS_NAME, DATASET_CHECKPOINT_NAME
from mt.trainer.models.pytransformer.transformer import TransformerModel
from mt.trainer.models.optim import ScheduledOptim
from mt.trainer.models.transformer.transformer import Transformer
from mt.trainer.tok import word_tokenizer


MODEL_NAME = "transformer"
WANDB_PROJECT = "nmt"  # Run "wandb login" in the terminal


MAX_EPOCHS = 50
LEARNING_RATE = 0.5e-3
BATCH_SIZE = 32 #int(32*1.5)
MAX_TOKENS = 4096 #int(4096*1.5)
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

wandb.init(project=WANDB_PROJECT, entity='salvacarrion')
config = wandb.config
config.model_name = MODEL_NAME
config.max_epochs = MAX_EPOCHS
config.learning_rate = LEARNING_RATE
config.batch_size = BATCH_SIZE
config.max_tokens = MAX_TOKENS
config.warmup_updates = WARMUP_UPDATES
config.patience = PATIENCE
config.acc_gradients = ACC_GRADIENTS
config.weight_decay = WEIGHT_DECAY
config.multigpu = MULTIGPU
config.device1 = str(DEVICE1)
config.device2 = str(DEVICE2)
config.num_workers = NUM_WORKERS
config.tok_model = TOK_MODEL
config.tok_size = TOK_SIZE
config.tok_folder = TOK_FOLDER
config.lowercase = LOWERCASE

###########################################################################
###########################################################################


def run_experiment(datapath, src, trg, model_name, domain=None):
    checkpoint_path = os.path.join(datapath, DATASET_CHECKPOINT_NAME, f"{model_name}_{domain}")

    # Load tokenizers
    src_tok, trg_tok = helpers.get_tokenizers(os.path.join(datapath, DATASET_TOK_NAME, TOK_FOLDER), src, trg, tok_model=TOK_MODEL, lower=LOWERCASE)

    # Load dataset
    train_ds = TranslationDataset(os.path.join(datapath, DATASET_CLEAN_NAME), src_tok, trg_tok, "train")
    val_ds = TranslationDataset(os.path.join(datapath, DATASET_CLEAN_NAME), src_tok, trg_tok, "val")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=lambda x: TranslationDataset.collate_fn(x, MAX_TOKENS), pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=lambda x: TranslationDataset.collate_fn(x, MAX_TOKENS), pin_memory=True)

    # Instantiate model #1
    model = Transformer(d_model=512,
                        enc_layers=6, dec_layers=6,
                        enc_heads=8, dec_heads=8,
                        enc_dff_dim=2048, dec_dff_dim=2048,
                        enc_dropout=0.1, dec_dropout=0.1,
                        max_src_len=2000, max_trg_len=2000,
                        src_tok=src_tok, trg_tok=trg_tok,
                        static_pos_emb=True).to(DEVICE1)
    print(f'The model has {model.count_parameters():,} trainable parameters')
    criterion = nn.CrossEntropyLoss(ignore_index=trg_tok.word2idx[trg_tok.PAD_WORD])
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Load weights
    #model.load_state_dict(torch.load(checkpoint_path + "_best.pt"))

    # Tensorboard (it needs some epochs to start working ~10-20)
    tb_writer = SummaryWriter(os.path.join(datapath, DATASET_LOGS_NAME, f"{model_name}"))
    wandb.watch(model)

    # Train and validate model
    fit(model, optimizer, train_loader=train_loader, val_loader=val_loader,
        epochs=MAX_EPOCHS, criterion=criterion,
        checkpoint_path=checkpoint_path,
        tb_writer=tb_writer)

    print("Done!")


def fit(model, optimizer, train_loader, val_loader, epochs, criterion, checkpoint_path, tb_writer=None):
    if not checkpoint_path:
        print("[WARNING] Training without a checkpoint path. The model won't be saved.")

    val_score_best = -1e9  # Loss 1e9; BLEU -1e9
    last_checkpoint = 0
    total_checkpoints = 0
    for epoch_i in range(epochs):
        start_time = time.time()

        # Train model
        tr_loss = train(model, optimizer, train_loader, criterion)

        # Evaluate
        val_loss, translations = evaluate(model, val_loader, criterion)

        # Log progress
        metrics = log_progress(epoch_i, start_time, tr_loss, val_loss, translations, tb_writer)

        # Save checkpoint
        if checkpoint_path:
            val_score = metrics["val"]["bleu"]
            if val_score > val_score_best:  # Loss <; BLEU >
                last_checkpoint = epoch_i
                val_score_best = val_score
                total_checkpoints += 1
                torch.save(model.state_dict(), checkpoint_path + "_best.pt")
                print("\t=> Checkpoint saved!")

            else:
                # Early stop
                if PATIENCE != -1 and (epoch_i - last_checkpoint) >= PATIENCE:
                    print(f"************************************************************************")
                    print(f"*** Early stop. Validation loss didn't improve for {PATIENCE} epochs ***")
                    print(f"************************************************************************")
                    break



def train(model, optimizer, data_loader, criterion, clip=1.0):
    epoch_loss = 0.0

    model.train()
    optimizer.zero_grad()
    for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
        try:
            # Get batch data
            src, src_mask, trg, trg_mask = [x.to(DEVICE1) for x in batch]
            batch_size, src_max_len, trg_max_len = src.shape[0], src.shape[1], trg.shape[1]

            # Get output
            optimizer.zero_grad()
            # output, _ = model(src, trg[:, :-1])
            output, _ = model(src, src_mask, trg[:, :-1], trg_mask[:, :-1])
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1).long()

            # Compute loss
            loss = criterion(output, trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            epoch_loss += loss.item()
        except RuntimeError as e:
            print("ERROR BATCH: " + str(i+1))
            print(e)

    return epoch_loss / len(data_loader)


def evaluate(model, data_loader, criterion):
    epoch_loss = 0
    srcs, trgs, pred_trgs = [], [], []

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
                pred_trgs += model.trg_tok.decode(output.argmax(2))
                trgs += model.trg_tok.decode(trg)
                srcs += model.src_tok.decode(src)
        except RuntimeError as e:
            print("ERROR BATCH: " + str(i+1))
            print(e)

    return epoch_loss / len(data_loader), (srcs, trgs, pred_trgs)


def log_progress(epoch_i, start_time, tr_loss, val_loss, translations=None, tb_writer=None):
    metrics = {
        "train": {
            "loss": tr_loss,
            "ppl": math.exp(tr_loss),
        },
        "val": {
            "loss": val_loss,
            "ppl": math.exp(val_loss),
        },
    }

    # Get additional metrics
    if translations:
        src_dec_all, hyp_dec_all, ref_dec_all = translations
        m_bleu_score = bleu_score([x.split(" ") for x in hyp_dec_all], [[x.split(" ")] for x in ref_dec_all])
        metrics["val"]["bleu"] = m_bleu_score

    # Print stuff
    end_time = time.time()
    epoch_mins, epoch_secs = helpers.epoch_time(start_time, end_time)
    print("------------------------------------------------------------")
    print(f'Epoch: {epoch_i + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\t- Train Loss: {metrics["train"]["loss"]:.3f} | Train PPL: {metrics["train"]["ppl"]:.3f}')
    print(f'\t- Val Loss: {metrics["val"]["loss"]:.3f} | Val PPL: {metrics["val"]["ppl"]:.3f} | Val BLEU: {metrics["val"]["bleu"]*100:.3f}')
    print("------------------------------------------------------------")

    # Tensorboard
    if tb_writer:
        for split in ["train", "val"]:
            for k, v in metrics[split].items():
                tb_writer.add_scalar(f'{split}_{k.lower()}', v, epoch_i+1)
                wandb.log({f'{split}_{k.lower()}': v})

    return metrics


if __name__ == "__main__":
    # Get all folders in the root path
    #datasets = [os.path.join(DATASETS_PATH, name) for name in os.listdir(DATASETS_PATH) if os.path.isdir(os.path.join(DATASETS_PATH, name))]
    datasets = [os.path.join(DATASETS_PATH, x) for x in ["health_es-en", "biological_es-en", "merged_es-en"]]
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
