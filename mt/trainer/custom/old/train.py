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
from mt import DATASETS_PATH,  DATASET_CLEAN_NAME, DATASET_TOK_NAME, DATASET_LOGS_NAME, DATASET_CHECKPOINT_NAME
from mt.trainer.models.pytransformer.transformer import TransformerModel
from mt.trainer.models.optim import ScheduledOptim
from mt.trainer.tok import fastbpe_tokenizer

MODEL_NAME = "transformer"
tok_folder = "bpe.16000"


MAX_EPOCHS = 50
LEARNING_RATE = 1e-3
BATCH_SIZE = int(32*0.5)
MAX_TOKENS = int(4096*0.5)
WARMUP_UPDATES = 4000
PATIENCE = 10
ACC_GRADIENTS = 8
WEIGHT_DECAY = 0.0001
WANDB_ENABLED = False
MULTIGPU = False
DEVICE1 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

if WANDB_ENABLED:
    wandb.init(project='nmt', entity='salvacarrion')
    config = wandb.config
    config.tok_folder = tok_folder
    config.learning_rate = LEARNING_RATE
    config.batch_size = BATCH_SIZE
    config.max_epochs = MAX_EPOCHS
    config.warmup_updates = WARMUP_UPDATES
    config.patience = PATIENCE
    config.acc_gradients = ACC_GRADIENTS
    config.weight_decay = WEIGHT_DECAY
    config.multigpu = MULTIGPU
    config.device1 = str(DEVICE1)
    config.device2 = str(DEVICE2)

###########################################################################
###########################################################################


def run_experiment(datapath, src, trg, model_name, tok_folder, domain=None, num_workers=0):
    checkpoint_path = os.path.join(datapath, DATASET_CHECKPOINT_NAME, f"{model_name}_{domain}_best.pt")

    # Load tokenizers
    src_tok, trg_tok = helpers.get_tokenizers(os.path.join(datapath, DATASET_TOK_NAME, tok_folder), src, trg, tok_model="fastbpe")

    # Load dataset
    datasets = helpers.load_dataset(os.path.join(datapath, DATASET_TOK_NAME, tok_folder), src, trg, splits=["train", "val", "test"])

    # Prepare data loaders
    train_loader = helpers.build_dataloader(datasets["train"], src_tok, trg_tok, fastbpe_tokenizer, batch_size=BATCH_SIZE, max_tokens=MAX_TOKENS, num_workers=num_workers)
    val_loader = helpers.build_dataloader(datasets["val"], src_tok, trg_tok, fastbpe_tokenizer, batch_size=BATCH_SIZE, max_tokens=MAX_TOKENS, num_workers=num_workers, shuffle=False)
    # test_loader = helpers.build_dataloader(datasets["test"], src_tok, trg_tok, batch_size=batch_size, max_tokens=max_tokens, num_workers=num_workers, shuffle=False)

    # Instantiate model #1
    model1 = TransformerModel(src_tok=src_tok, trg_tok=trg_tok)
    model1.to(DEVICE1)
    # model1.load_state_dict(torch.load(checkpoint_path))
    optimizer1 = ScheduledOptim(
        optim.Adam(model1.parameters(), betas=(0.9, 0.98), eps=1e-09, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY),
        model1.d_model, WARMUP_UPDATES)

    if MULTIGPU and torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model1 = nn.DataParallel(model1)

    if WANDB_ENABLED:
        wandb.watch(model1)

    # Set loss (ignore when the target token is <pad>)
    criterion = nn.CrossEntropyLoss(ignore_index=trg_tok.word2idx[trg_tok.PAD_WORD])

    # Tensorboard (it needs some epochs to start working ~10-20)
    tr_writer = SummaryWriter(os.path.join(datapath, DATASET_LOGS_NAME, f"{model_name}/train"))
    val_writer = SummaryWriter(os.path.join(datapath, DATASET_LOGS_NAME, f"{model_name}/val"))

    # Train and validate model
    fit((model1, optimizer1), (None, None), train_loader=train_loader, val_loader=val_loader,
        epochs=MAX_EPOCHS, criterion=criterion,
        checkpoint_path=checkpoint_path,
        tr_writer=tr_writer, val_writer=val_writer)

    print("Done!")


def fit(model_opt1, model_opt2, train_loader, val_loader, epochs, criterion, checkpoint_path, tr_writer=None, val_writer=None):
    if not checkpoint_path:
        print("[WARNING] Traning without checkpoint path. The model won't be saved.")

    lowest_val = 1e9
    last_checkpoint = 0
    for epoch_i in range(epochs):
        # Train model
        tr_loss, tr_metrics = train(model_opt1, model_opt2, train_loader, criterion, epoch_i=epoch_i, tb_writer=tr_writer)

        # Evaluate
        val_loss, val_metrics = evaluate(model_opt1[0], val_loader, criterion, epoch_i=epoch_i, tb_writer=val_writer)

        # Save checkpoint
        if checkpoint_path:
            if val_loss < lowest_val:
                avg_bleu = sum([x["torch_bleu"] for x in val_metrics]) / len(val_metrics)
                print(f"New best score! Loss={val_loss} | BLEU={avg_bleu}. (Saving checkpoint...)")
                last_checkpoint = epoch_i
                lowest_val = val_loss
                torch.save(model_opt1[0].state_dict(), checkpoint_path)
                print("=> Checkpoint saved!")

            else:
                # Early stop
                if PATIENCE != -1 and (epoch_i-last_checkpoint) >= PATIENCE:
                    print(f"Early stop. Validation loss didn't improve for {PATIENCE} epochs")
                    break


def train(model_opt1, model_opt2, data_loader, criterion, clip=1.0, log_interval=1, epoch_i=None, tb_writer=None):
    total_loss = 0.0
    all_metrics = []
    start_time = time.time()

    # Unpack values
    (model1, optimizer1) = model_opt1
    (model2, optimizer2) = model_opt2

    model1.train()
    optimizer1.zero_grad()

    loss = 0
    for i, batch in enumerate(data_loader):
        # Get batch data
        src1, src_mask1, trg1, trg_mask1 = [x.to(DEVICE1) for x in batch]
        batch_size, src_max_len, trg_max_len = src1.shape[0], src1.shape[1], trg1.shape[1]

        # Create a padding mask (no-padded=0, padded=1)
        src_key_padding_mask = ~src_mask1.type(torch.bool).to(DEVICE1)
        tgt_key_padding_mask = ~trg_mask1.type(torch.bool).to(DEVICE1)
        memory_key_padding_mask = src_key_padding_mask.clone()  # the src_mask used in the decoder

        # Create tgt_inp and tgt_out (which is tgt_inp but shifted by 1)
        tgt_inp, tgt_out = trg1[:, :-1], trg1[:, 1:]

        # Get output
        output1 = model1(src1, tgt_inp, src_key_padding_mask, tgt_key_padding_mask[:, :-1], memory_key_padding_mask)

        # Compute backward
        loss = criterion(rearrange(output1, 'b t v -> (b t) v'), rearrange(tgt_out, 'b o -> (b o)'))
        loss /= ACC_GRADIENTS
        loss.backward()
        total_loss += loss.item()

        # Accumulate gradients
        if (i+1) % ACC_GRADIENTS == 0 or (i+1) == len(data_loader):
            # Clip params
            torch.nn.utils.clip_grad_norm_(model1.parameters(), clip)

            # Update parameters
            optimizer1.step_and_update_lr()
            optimizer1.zero_grad()

        # Log progress
        if (i+1) % log_interval == 0:
            metrics = log_progress("train", total_loss, epoch_i+1, i+1, len(data_loader), start_time, tb_writer)
            all_metrics.append(metrics)

    return total_loss / len(data_loader), all_metrics


def evaluate(model, data_loader, criterion, log_interval=1, epoch_i=None, tb_writer=None, print_translations=True):
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
            outputs = model(src, tgt_inp, src_key_padding_mask, tgt_key_padding_mask[:, :-1], memory_key_padding_mask)
            loss = criterion(rearrange(outputs, 'b t v -> (b t) v'), rearrange(tgt_out, 'b o -> (b o)'))

            total_loss += loss.item()

            # Log progress
            if (i+1) % log_interval == 0:
                # Get translations
                src_dec, ref_dec, hyp_dec = get_translations(src, trg, torch.argmax(outputs.detach(), dim=-1), model.src_tok, model.trg_tok)
                if print_translations:
                    src_dec_all += src_dec
                    ref_dec_all += ref_dec
                    hyp_dec_all += hyp_dec

                # Log progress
                metrics = log_progress("val", total_loss, epoch_i+1, i+1, len(data_loader), start_time, tb_writer, translations=(src_dec, ref_dec, hyp_dec))
                all_metrics.append(metrics)

    # Print translations
    if print_translations:
        helpers.print_translations(hypothesis=hyp_dec_all, references=ref_dec_all, source=src_dec_all, limit=50)

    return total_loss / len(data_loader), all_metrics


def log_progress(prefix, total_loss, epoch_i, batch_i, n_batches, start_time, tb_writer, translations=None):
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
            if WANDB_ENABLED:
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
        run_experiment(dataset, src, trg, model_name=MODEL_NAME, tok_folder=tok_folder, domain=domain)
