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
from torchsummary import summary

import sacrebleu
from datasets import load_metric
from torchtext.data.metrics import bleu_score

from tqdm import tqdm

from mt import utils
from mt import helpers
from mt.dataloaders.datasets import TranslationDataset

from mt import DATASETS_PATH, DATASET_CLEAN_NAME, DATASET_CLEAN_SORTED_NAME, DATASET_TOK_NAME, DATASET_LOGS_NAME, DATASET_CHECKPOINT_NAME
from mt.trainer.models.transformer.transformer import Transformer
from mt.trainer.models.transformer.transformer_dyn import TransformerDyn

from torch.utils.data.sampler import SequentialSampler
from torchnlp.samplers import BucketBatchSampler
from mt.dataloaders.max_tokens_batch_sampler import MaxTokensBatchSampler


def len_func(ds, i):
    return len(ds.datasets.iloc[i]["src"].split())


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


def freeze_layers(model):
    for param in model.parameters():
        param.requires_grad = False


def save_checkpoint(model, checkpoint_path, metrics, best_score):
    # Save checkpoint
    if checkpoint_path:
        # Save last
        torch.save(model.state_dict(), checkpoint_path + "_last.pt")

        # Save best BLEU
        score = metrics["val"]["sacrebleu_bleu"]
        if score > best_score:  # Loss <; BLEU >
            best_score = score
            torch.save(model.state_dict(), checkpoint_path + "_best.pt")
            print("\t=> Checkpoint saved!")
    return best_score


def evaluate(model, data_loader, criterion, device):
    epoch_loss = 0
    src_dec_all, hyp_dec_all, ref_dec_all = [], [], []

    model.eval()
    for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
        try:
            with torch.no_grad():
                # Get batch data
                src, src_mask, trg, trg_mask = [x.to(device) for x in batch]

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


def log_progress(epoch_i, start_time, tr_loss=None, val_loss=None, tb_writer=None, translations=None, print_translations=True, prefix=None, **kwargs):
    metrics = {}
    if tr_loss:
        metrics["train"] = {
            "loss": tr_loss,
            "ppl": math.exp(tr_loss),
        }

    if val_loss:
        metrics["val"] = {
            "loss": val_loss,
            "ppl": math.exp(val_loss),
        }

    # Get additional metrics
    if translations:
        src_dec_all, hyp_dec_all, ref_dec_all = translations

        if val_loss:
            val_metrics = compute_metrics(hyp_dec_all, ref_dec_all, **kwargs)
            metrics["val"].update(val_metrics)

        # Print translations
        if print_translations:
            helpers.print_translations(hyp_dec_all, ref_dec_all, src_dec_all, limit=50)

    # Print stuff
    end_time = time.time()
    epoch_hours, epoch_mins, epoch_secs = helpers.epoch_time(start_time, end_time)
    print("------------------------------------------------------------")
    print(f'Epoch: {epoch_i + 1:02} | Time: {epoch_mins}m {epoch_secs}s | [Prefix: {prefix}]')
    if tr_loss:
        print(f'\t- Train Loss: {metrics["train"]["loss"]:.3f} | Train PPL: {metrics["train"]["ppl"]:.3f}')
    if val_loss:
        extra_metrics = [f"Val {k.lower()}: {v:.3f}" for k, v in metrics["val"].items() if k not in {"loss", "ppl"}]
        print(f'\t- Val Loss: {metrics["val"]["loss"]:.3f} | Val PPL: {metrics["val"]["ppl"]:.3f} | ' + " | ".join(extra_metrics))
    print("------------------------------------------------------------")

    # Tensorboard
    if tb_writer:
        prefix = f"{prefix}_" if prefix else ""
        for split in list(metrics.keys()):
            for k, v in metrics[split].items():
                tb_writer.add_scalar(f'{prefix}{split}_{k.lower()}', v, epoch_i+1)
                wandb.log({f'{prefix}{split}_{k.lower()}': v})

    return metrics


def compute_metrics(hyp_dec_all, ref_dec_all, use_sacrebleu=True, use_torchtext=True, use_ter=False):
    metrics = {}

    # Sacrebleu
    if use_sacrebleu:
        metrics["sacrebleu_rawcorpusbleu"] = sacrebleu.raw_corpus_bleu(hyp_dec_all, [ref_dec_all]).score
        metrics["sacrebleu_bleu"] = sacrebleu.corpus_bleu(hyp_dec_all, [ref_dec_all]).score
        metrics["sacrebleu_chrf"] = sacrebleu.corpus_chrf(hyp_dec_all, [ref_dec_all]).score
        if use_ter:  # Quite slow
            metrics["sacrebleu_ter"] = sacrebleu.corpus_ter(hyp_dec_all, [ref_dec_all]).score

    # Torchtext
    if use_torchtext:
        m_bleu_score = bleu_score([x.split(" ") for x in hyp_dec_all], [[x.split(" ")] for x in ref_dec_all])
        metrics["torchtext_bleu"] = m_bleu_score * 100
    return metrics


def get_translations(data, model, device, max_length=50, beam_width=3):
    src_dec_all, hyp_dec_all, ref_dec_all = [], [], []

    for batch in tqdm(data, total=len(data)):
        # Get batch data
        src, src_mask, trg, trg_mask = [x.to(device) for x in batch]
        batch_size, src_max_len, trg_max_len = src.shape[0], src.shape[1], trg.shape[1]

        smart_max_length = min(max_length, int(trg_max_len*1.2))  # trick: We shouldn't know the max_length
        pred_trg, probs, _ = model.translate_batch(src, max_length=smart_max_length, beam_width=beam_width)
        pred_trg = pred_trg[0]  # Get first candidate

        # cut off <eos> token
        hyp_dec_all += model.trg_tok.decode(pred_trg, remove_special_tokens=True)
        ref_dec_all += model.trg_tok.decode(trg, remove_special_tokens=True)
        src_dec_all += model.src_tok.decode(src, remove_special_tokens=True)

    return src_dec_all, hyp_dec_all, ref_dec_all


def get_data_loader(sampler_name, dataset, batch_size, max_tokens, num_workers, shuffle):
    kwargs_test = {}
    if sampler_name == "bucket":
        sampler = BucketBatchSampler(SequentialSampler(dataset), batch_size=batch_size, drop_last=False,
                                          sort_key=lambda i: len(dataset.datasets.iloc[i]["src"].split()))
    elif sampler_name == "maxtokens":
        sampler = MaxTokensBatchSampler(SequentialSampler(dataset), shuffle=shuffle, batch_size=batch_size,
                                             max_tokens=max_tokens, drop_last=False,
                                             sort_key=lambda i: len(dataset.datasets.iloc[i]["src"].split()))
    else:
        sampler = None
        kwargs_test = {"batch_size": batch_size, "shuffle": shuffle}

    # Define dataloader
    data_loader = DataLoader(dataset, num_workers=num_workers,
                             collate_fn=lambda x: TranslationDataset.collate_fn(x, max_tokens), pin_memory=True,
                             batch_sampler=sampler, **kwargs_test)
    return data_loader
