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

MODEL_NAME = "transformer_bv"


MAX_EPOCHS = 50
LEARNING_RATE = 0.0005 #1e-3
BATCH_SIZE = 128 #int(32*1.5)
MAX_TOKENS = int(4096*1.5)
WARMUP_UPDATES = 4000
PATIENCE = 10
ACC_GRADIENTS = 1
WEIGHT_DECAY = 0.0001
MULTIGPU = False
DEVICE1 = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # torch.device("cpu") #
DEVICE2 = None  #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0
TOK_MODEL = "fastbpe"
TOK_FOLDER = "bpe.16000"

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

# wandb.init(project='nmt', entity='salvacarrion')
# config = wandb.config
# config.tok_folder = tok_folder
# config.learning_rate = LEARNING_RATE
# config.batch_size = BATCH_SIZE
# config.max_epochs = MAX_EPOCHS
# config.warmup_updates = WARMUP_UPDATES
# config.patience = PATIENCE
# config.acc_gradients = ACC_GRADIENTS
# config.weight_decay = WEIGHT_DECAY
# config.multigpu = MULTIGPU
# config.device1 = str(DEVICE1)
# config.device2 = str(DEVICE2)

###########################################################################
###########################################################################


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


def run_experiment(datapath, src, trg, model_name, domain=None):
    checkpoint_path = os.path.join(datapath, DATASET_CHECKPOINT_NAME, f"{model_name}_{domain}_best.pt")

    # Load tokenizers
    src_tok, trg_tok = helpers.get_tokenizers(os.path.join(datapath, DATASET_TOK_NAME, TOK_FOLDER), src, trg, tok_model=TOK_MODEL)

    # Load dataset
    train_ds = TranslationDataset(os.path.join(datapath, DATASET_CLEAN_NAME), src_tok, trg_tok, "train")
    val_ds = TranslationDataset(os.path.join(datapath, DATASET_CLEAN_NAME), src_tok, trg_tok, "val")
    train_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=TranslationDataset.collate_fn, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=TranslationDataset.collate_fn, pin_memory=True)

    # Instantiate model #1
    model = Transformer(d_model=256,
                        enc_layers=3, dec_layers=3,
                        enc_heads=8, dec_heads=8,
                        enc_dff_dim=512, dec_dff_dim=512,
                        enc_dropout=0.1, dec_dropout=0.1,
                        max_src_len=2000, max_trg_len=2000,
                        src_tok=src_tok, trg_tok=trg_tok)
    model.to(DEVICE1)
    model.apply(initialize_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print(f'The model has {count_parameters(model):,} trainable parameters')
    criterion = nn.CrossEntropyLoss(ignore_index=trg_tok.word2idx[trg_tok.PAD_WORD])

    # Tensorboard (it needs some epochs to start working ~10-20)
    tb_writer = SummaryWriter(os.path.join(datapath, DATASET_LOGS_NAME, f"{model_name}"))
    # wandb.watch(model1)

    # Train and validate model
    fit(model, optimizer, train_loader=train_loader, val_loader=val_loader,
        epochs=MAX_EPOCHS, criterion=criterion,
        checkpoint_path=checkpoint_path,
        tb_writer=tb_writer)

    print("Done!")


def fit(model, optimizer, train_loader, val_loader, epochs, criterion, checkpoint_path, tb_writer=None):
    if not checkpoint_path:
        print("[WARNING] Traning without checkpoint path. The model won't be saved.")

    lowest_val = 1e9
    last_checkpoint = 0
    for epoch_i in range(epochs):
        start_time = time.time()

        # Train model
        tr_loss = train(model, optimizer, train_loader, criterion)

        # Evaluate
        val_loss = evaluate(model, val_loader, criterion)

        # Log progress
        log_progress(epoch_i, start_time, tr_loss, val_loss)

        # Save checkpoint
        if checkpoint_path:
            if val_loss < lowest_val:
                last_checkpoint = epoch_i
                lowest_val = val_loss
                torch.save(model.state_dict(), checkpoint_path)
                print("\t=> Checkpoint saved!")

            else:
                # Early stop
                if PATIENCE != -1 and (epoch_i-last_checkpoint) >= PATIENCE:
                    print(f"****************************************************************")
                    print(f"Early stop. Validation loss didn't improve for {PATIENCE} epochs")
                    print(f"****************************************************************")
                    break


def train(model, optimizer, data_loader, criterion, clip=1.0):
    epoch_loss = 0.0

    model.train()
    optimizer.zero_grad()
    for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
        # Get batch data
        src, src_mask, trg, trg_mask = [x.to(DEVICE1) for x in batch]
        batch_size, src_max_len, trg_max_len = src.shape[0], src.shape[1], trg.shape[1]

        # Get output
        optimizer.zero_grad()
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

    return epoch_loss / len(data_loader)


def evaluate(model, data_loader, criterion):
    epoch_loss = 0

    model.eval()
    for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
        with torch.no_grad():
            # Get batch data
            src, src_mask, trg, trg_mask = [x.to(DEVICE1) for x in batch]

            # Get output
            output, _ = model(src, src_mask, trg[:, :-1], trg_mask[:, :-1])
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)

            # Compute loss
            loss = criterion(output, trg.long())
            epoch_loss += loss.item()

    return epoch_loss / len(data_loader)


def log_progress(epoch_i, start_time, tr_loss, val_loss, tb_writer=None):
    metrics = {
        "train": {
            "train_loss": tr_loss,
            "train_ppl": math.exp(tr_loss),
        },
        "val": {
            "val_loss": val_loss,
            "val_ppl": math.exp(val_loss),
        },
    }

    # Print stuff
    end_time = time.time()
    epoch_mins, epoch_secs = helpers.epoch_time(start_time, end_time)
    print("------------------------------------------------------------")
    print(f'Epoch: {epoch_i + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\t- Train Loss: {tr_loss:.3f} | Train PPL: {math.exp(tr_loss):7.3f}')
    print(f'\t- Val Loss: {tr_loss:.3f} | Val PPL: {math.exp(tr_loss):7.3f}')
    print("------------------------------------------------------------")

    # Tensorboard
    if tb_writer:
        for split in ["train", "val"]:
            for k, v in metrics.items():
                tb_writer.add_scalar(f'{split}_{k.lower()}', v, epoch_i)
                # wandb.log({f'{split}_{k.lower()}': v})

    return metrics



if __name__ == "__main__":
    # Get all folders in the root path
    #datasets = [os.path.join(DATASETS_PATH, name) for name in os.listdir(DATASETS_PATH) if os.path.isdir(os.path.join(DATASETS_PATH, name))]
    datasets = [os.path.join(DATASETS_PATH, "multi30k_de-en")]
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
