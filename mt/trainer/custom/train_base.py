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

import sacrebleu
from datasets import load_metric
from torchtext.data.metrics import bleu_score

from tqdm import tqdm

from mt import utils
from mt import helpers
from mt.dataloaders.datasets import TranslationDataset
from mt.trainer.custom import base

from mt import DATASETS_PATH, DATASET_CLEAN_NAME, DATASET_CLEAN_SORTED_NAME, DATASET_TOK_NAME, DATASET_LOGS_NAME, DATASET_CHECKPOINT_NAME
from mt.trainer.models.transformer.transformer import Transformer

from torch.utils.data.sampler import SequentialSampler
from torchnlp.samplers import BucketBatchSampler
from mt.dataloaders.max_tokens_batch_sampler import MaxTokensBatchSampler

MODEL_NAME = "transformer"
WANDB_PROJECT = "nmt"  # Run "wandb login" in the terminal

MAX_EPOCHS = 50
LEARNING_RATE = 0.5e-3
BATCH_SIZE = 128 #int(32*1.5)
MAX_TOKENS = 1024 #int(4096*1.5)
WARMUP_UPDATES = 4000
PATIENCE = 10
ACC_GRADIENTS = 1  # Tricky. It can hurt the training.
WEIGHT_DECAY = 0.0001
CLIP_GRADIENTS = 1.0
MULTIGPU = False
DEVICE1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # torch.device("cpu") #
DEVICE2 = None  #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 8
TOK_MODEL = "bpe"
TOK_SIZE = 16000
TOK_FOLDER = f"{TOK_MODEL}.{TOK_SIZE}"
LOWERCASE = False
TRUNCATE = True
MAX_LENGTH_TRUNC = 2000
SAMPLER_NAME = "maxtokens" #"maxtokens"  # bucket # None
START_FROM_CHECKPOINT = "transformer_health_best.pt"

print(f"Device #1: {DEVICE1}")
print(f"Device #2: {DEVICE2}")
print(f"CUDA devices count: {torch.cuda.device_count()}")

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


def run_experiment(datapath, src, trg, model_name, domain=None):
    start_time = time.time()

    ###########################################################################
    ###########################################################################

    wandb.init(project=WANDB_PROJECT, entity='salvacarrion', reinit=True)
    config = wandb.config
    config.model_name = MODEL_NAME
    config.domain = domain
    config.max_epochs = MAX_EPOCHS
    config.learning_rate = LEARNING_RATE
    config.batch_size = BATCH_SIZE
    config.max_tokens = MAX_TOKENS
    config.warmup_updates = WARMUP_UPDATES
    config.patience = PATIENCE
    config.acc_gradients = ACC_GRADIENTS
    config.weight_decay = WEIGHT_DECAY
    config.clip_gradients = CLIP_GRADIENTS
    config.multigpu = MULTIGPU
    config.device1 = str(DEVICE1)
    config.device2 = str(DEVICE2)
    config.num_workers = NUM_WORKERS
    config.tok_model = TOK_MODEL
    config.tok_size = TOK_SIZE
    config.tok_folder = TOK_FOLDER
    config.lowercase = LOWERCASE
    config.truncate = TRUNCATE
    config.max_length_truncate = MAX_LENGTH_TRUNC
    config.sampler_name = str(SAMPLER_NAME)
    config.start_from_checkpoint = START_FROM_CHECKPOINT
    print(config)
    ###########################################################################
    ###########################################################################

    checkpoint_path = os.path.join(datapath, DATASET_CHECKPOINT_NAME, f"{model_name}_{domain}")

    # Load tokenizers
    src_tok, trg_tok = helpers.get_tokenizers(os.path.join(datapath, DATASET_TOK_NAME, TOK_FOLDER), src, trg,
                                              tok_model=TOK_MODEL, lower=LOWERCASE, truncation=TRUNCATE, max_length=MAX_LENGTH_TRUNC)

    # Load dataset
    datapath_clean = DATASET_CLEAN_NAME
    if TOK_MODEL == "bpe":  # Do not preprocess again when using bpe
        src_tok.apply_bpe = False
        trg_tok.apply_bpe = False
        datapath_clean = os.path.join(DATASET_TOK_NAME, TOK_FOLDER)

    # Get datasets
    train_ds = TranslationDataset(os.path.join(datapath, datapath_clean), src_tok, trg_tok, "train")
    val_ds = TranslationDataset(os.path.join(datapath, datapath_clean), src_tok, trg_tok, "val")

    # Get dataloaders
    train_loader = base.get_data_loader(SAMPLER_NAME, train_ds, BATCH_SIZE, MAX_TOKENS, NUM_WORKERS, shuffle=True)
    val_loader = base.get_data_loader(SAMPLER_NAME, val_ds, BATCH_SIZE, MAX_TOKENS, NUM_WORKERS, shuffle=False)

    # Instantiate model #1
    model = Transformer(d_model=256,
                        enc_layers=3, dec_layers=3,
                        enc_heads=8, dec_heads=8,
                        enc_dff_dim=512, dec_dff_dim=512,
                        enc_dropout=0.1, dec_dropout=0.1,
                        max_src_len=2000, max_trg_len=2000,
                        src_tok=src_tok, trg_tok=trg_tok,
                        static_pos_emb=True).to(DEVICE1)
    model.apply(base.initialize_weights)
    print(f'The model has {model.count_parameters():,} trainable parameters')
    criterion = nn.CrossEntropyLoss(ignore_index=trg_tok.word2idx[trg_tok.PAD_WORD])
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Load weights
    if START_FROM_CHECKPOINT:
        from_checkpoint_path = os.path.join(datapath, DATASET_CHECKPOINT_NAME, START_FROM_CHECKPOINT)
        print(f"Loading weights from: {from_checkpoint_path}")
        model.load_state_dict(torch.load(from_checkpoint_path))

    # Tensorboard (it needs some epochs to start working ~10-20)
    tb_writer = SummaryWriter(os.path.join(datapath, DATASET_LOGS_NAME, f"{model_name}"))
    wandb.watch(model)

    # Train and validate model
    fit(model, optimizer, train_loader=train_loader, val_loader=val_loader,
        epochs=MAX_EPOCHS, criterion=criterion,
        checkpoint_path=checkpoint_path,
        tb_writer=tb_writer)

    print("************************************************************")
    epoch_hours, epoch_mins, epoch_secs = helpers.epoch_time(start_time, end_time=time.time())
    print(f'Time experiment: {epoch_hours}h {epoch_mins}m {epoch_secs}s')
    print("************************************************************")
    print("Done!")


def fit(model, optimizer, train_loader, val_loader, epochs, criterion, checkpoint_path, tb_writer=None):
    if not checkpoint_path:
        print("[WARNING] Training without a checkpoint path. The model won't be saved.")

    best_score = -1e9  # Loss 1e9; BLEU -1e9
    last_checkpoint = 0
    for epoch_i in range(epochs):
        start_time = time.time()

        # Train model
        tr_loss = train(model, optimizer, train_loader, criterion)

        # Evaluate
        val_loss, translations = base.evaluate(model, val_loader, criterion, device=DEVICE1)

        # Log progress
        metrics = base.log_progress(epoch_i=epoch_i, start_time=start_time, tr_loss=tr_loss, val_loss=val_loss,
                                    tb_writer=tb_writer, translations=translations, print_translations=True, prefix=None)

        # Checkpoint
        new_best_score = base.save_checkpoint(model, checkpoint_path, metrics, best_score)
        last_checkpoint = epoch_i if best_score != new_best_score else last_checkpoint
        best_score = new_best_score

        # Early stop
        if PATIENCE != -1 and (epoch_i - last_checkpoint) >= PATIENCE:
            print(f"************************************************************************")
            print(f"*** Early stop. Validation loss didn't improve for {PATIENCE} epochs ***")
            print(f"************************************************************************")
            break


def train(model, optimizer, data_loader, criterion):
    epoch_loss = 0.0

    model.train()
    optimizer.zero_grad()
    for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
        try:
            # Get batch data
            src, src_mask, trg, trg_mask = [x.to(DEVICE1) for x in batch]
            batch_size, src_max_len, trg_max_len = src.shape[0], src.shape[1], trg.shape[1]

            # Get output
            output, _ = model(src, src_mask, trg[:, :-1], trg_mask[:, :-1])
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1).long()

            # Compute loss
            loss = criterion(output, trg) / ACC_GRADIENTS  # Normalize loss
            loss.backward()

            # Track total loss
            epoch_loss += loss.item()

            # Accumulate gradients
            if (i+1) % ACC_GRADIENTS == 0 or (i+1) == len(data_loader):
                # Clip gradients
                if CLIP_GRADIENTS > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRADIENTS)
                optimizer.step()
                optimizer.zero_grad()
        except RuntimeError as e:
            print("ERROR BATCH: " + str(i+1))
            print(e)

    return epoch_loss / len(data_loader)


if __name__ == "__main__":
    # Get all folders in the root path
    datasets = [os.path.join(DATASETS_PATH, x) for x in ["health_es-en", "biological_es-en", "merged_es-en"]]
    # datasets = [os.path.join(DATASETS_PATH, x) for x in ["health_biological_es-en"]]
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
