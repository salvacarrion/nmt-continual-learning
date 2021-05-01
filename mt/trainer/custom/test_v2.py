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
from torchtext.data.metrics import bleu_score

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
from mt import DATASETS_PATH, DATASET_CLEAN_NAME, DATASET_TOK_NAME, DATASET_LOGS_NAME, DATASET_CHECKPOINT_NAME, DATASET_EVAL_NAME
from mt.trainer.models.pytransformer.transformer import TransformerModel
from mt.trainer.models.optim import ScheduledOptim
from mt.trainer.models.transformer.transformer import Transformer
from mt.trainer.tok import word_tokenizer


MODEL_NAME = "transformer"


MAX_EPOCHS = 50
LEARNING_RATE = 0.5e-3
BATCH_SIZE = 32 #int(32*1.5)
MAX_TOKENS = 4096  #int(4096*1.5)
WARMUP_UPDATES = 4000
PATIENCE = 10
ACC_GRADIENTS = 1
WEIGHT_DECAY = 0.0001
MULTIGPU = False
DEVICE1 = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # torch.device("cpu") #
DEVICE2 = None  #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0
TOK_MODEL = "wt"
TOK_SIZE = 16000
TOK_FOLDER = f"{TOK_MODEL}.{TOK_SIZE}"
LOWERCASE = True

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


def run_experiment(datapath, src, trg, model_name, domain=None):
    # checkpoint_path = os.path.join(datapath, DATASET_CHECKPOINT_NAME, f"{model_name}_{domain}_best.pt")
    checkpoint_path = os.path.join(datapath, DATASET_CHECKPOINT_NAME, "transformer_health_best.pt")

    # Load tokenizers
    src_tok, trg_tok = helpers.get_tokenizers(os.path.join(datapath, DATASET_TOK_NAME, TOK_FOLDER), src, trg, tok_model=TOK_MODEL, lower=LOWERCASE)

    # Load dataset
    test_ds = TranslationDataset(os.path.join(datapath, DATASET_CLEAN_NAME), src_tok, trg_tok, "test")
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=lambda x: TranslationDataset.collate_fn(x, MAX_TOKENS), pin_memory=True)

    # Instantiate model #1
    model = Transformer(d_model=512//2,
                        enc_layers=6//2, dec_layers=6//2,
                        enc_heads=8, dec_heads=8,
                        enc_dff_dim=2048//2, dec_dff_dim=2048//2,
                        enc_dropout=0.1, dec_dropout=0.1,
                        max_src_len=2000, max_trg_len=2000,
                        src_tok=src_tok, trg_tok=trg_tok,
                        static_pos_emb=True).to(DEVICE1)
    print(f'The model has {model.count_parameters():,} trainable parameters')
    criterion = nn.CrossEntropyLoss(ignore_index=trg_tok.word2idx[trg_tok.PAD_WORD])

    # Load weights
    model.load_state_dict(torch.load(checkpoint_path))

    # Evaluate
    start_time = time.time()
    val_loss, translations = evaluate(model, test_loader, criterion)

    # Log progress
    metrics = log_progress(start_time, val_loss, translations)

    # Get translations
    src_dec_all, hyp_dec_all, ref_dec_all = get_translations(model, test_loader, src_tok, trg_tok)

    # Print translations
    helpers.print_translations(hyp_dec_all, ref_dec_all, src_dec_all, limit=50)

    # Compute scores
    m_bleu_score = bleu_score([x.split(" ") for x in hyp_dec_all], [[x.split(" ")] for x in ref_dec_all])
    print(f'BLEU score = {m_bleu_score*100:.3f}')

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
    print("Translations written!")


def evaluate(model, data_loader, criterion):
    epoch_loss = 0
    src_dec_all, hyp_dec_all, ref_dec_all = [], [], []

    model.eval()
    for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
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
            hyp_dec_all += model.trg_tok.decode(output.argmax(2))
            ref_dec_all += model.trg_tok.decode(trg)
            src_dec_all += model.src_tok.decode(src)

    return epoch_loss / len(data_loader), (src_dec_all, hyp_dec_all, ref_dec_all)


def log_progress(start_time, val_loss, translations=None):
    metrics = {
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

        # Print translations
        helpers.print_translations(hyp_dec_all, ref_dec_all, src_dec_all, limit=50)

    # Print stuff
    end_time = time.time()
    epoch_mins, epoch_secs = helpers.epoch_time(start_time, end_time)
    print("------------------------------------------------------------")
    print(f'Evaluate | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\t- Val Loss: {metrics["val"]["loss"]:.3f} | Val PPL: {metrics["val"]["ppl"]:.3f} | Val BLEU: {metrics["val"]["bleu"]*100:.3f}')
    print("------------------------------------------------------------")

    return metrics


def get_translations(model, data_loader, src_tok, trg_tok, max_len=100):
    srcs, trgs, pred_trgs = [], [], []

    model.eval()
    for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
        # Get batch data
        src, src_mask, trg, trg_mask = [x.to(DEVICE1) for x in batch]
        batch_size, src_max_len, trg_max_len = src.shape[0], src.shape[1], trg.shape[1]

        pred_trg, _ = translate_sentence_vectorized(src, src_tok, trg_tok, model, max_len)

        pred_trgs += trg_tok.decode(pred_trg)
        trgs += trg_tok.decode(trg)
        srcs += src_tok.decode(src)
        break

    return srcs, pred_trgs, trgs


def translate_sentence_vectorized(src_tensor, src_tok, trg_tok, model, max_len):
    assert isinstance(src_tensor, torch.Tensor)

    model.eval()
    src_mask = model.make_src_mask(src_tensor)

    with torch.no_grad():
        enc_src = model.enc_input(src_tensor)
        enc_src = model.encoder(enc_src, src_mask)
    # enc_src = [batch_sz, src_len, hid_dim]

    TRG_SOS_IDX = trg_tok.word2idx[trg_tok.SOS_WORD]
    TRG_EOS_IDX = trg_tok.word2idx[trg_tok.EOS_WORD]
    trg_indexes = [[TRG_SOS_IDX] for _ in range(len(src_tensor))]
    # Even though some examples might have been completed by producing a <eos> token
    # we still need to feed them through the model because other are not yet finished
    # and all examples act as a batch. Once every single sentence prediction encounters
    # <eos> token, then we can stop predicting.
    translations_done = [0] * len(src_tensor)
    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).to(enc_src.device)
        trg_mask = model.make_trg_mask(trg_tensor)
        with torch.no_grad():
            trg_tensor = model.dec_input(trg_tensor)
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
            output = model.fc_out(output)  # (B, L, d_model) => (B, L, vocab)

        pred_tokens = output.argmax(2)[:,-1]
        for i, pred_token_i in enumerate(pred_tokens):
            trg_indexes[i].append(pred_token_i)
            if pred_token_i == TRG_EOS_IDX:
                translations_done[i] = 1
        if all(translations_done):
            break

    # Iterate through each predicted example one by one;
    # Cut-off the portion including the after the <eos> token
    pred_sentences = []
    for trg_sentence in trg_indexes:
        pred_sentence = []
        for i in range(1, len(trg_sentence)):
            trg_idx = int(trg_sentence[i])
            pred_sentence.append(trg_idx)
            if trg_idx == TRG_EOS_IDX:
                break
        pred_sentences.append(pred_sentence)

    return pred_sentences, attention


def calculate_bleu_alt(iterator, src_field, trg_field, model, device, max_len = 50):
    trgs = []
    pred_trgs = []
    with torch.no_grad():
        for batch in iterator:
            src = batch.src
            trg = batch.trg
            _trgs = []
            for sentence in trg:
                tmp = []
                # Start from the first token which skips the <start> token
                for i in sentence[1:]:
                    # Targets are padded. So stop appending as soon as a padding or eos token is encountered
                    if i == trg_field.vocab.stoi[trg_field.eos_token] or i == trg_field.vocab.stoi[trg_field.pad_token]:
                        break
                    tmp.append(trg_field.vocab.itos[i])
                _trgs.append([tmp])
            trgs += _trgs
            pred_trg, _ = translate_sentence_vectorized(src, src_field, trg_field, model, device)
            pred_trgs += pred_trg
    return pred_trgs, trgs, bleu_score(pred_trgs, trgs)


if __name__ == "__main__":
    # Get all folders in the root path
    datasets = [os.path.join(DATASETS_PATH, x) for x in ["health_es-en", "biological_es-en", "merged_es-en"]]
    # datasets = [os.path.join(DATASETS_PATH, "multi30k_de-en")]
    for dataset in datasets:
        domain, (src, trg) = utils.get_dataset_ids(dataset)
        fname_base = f"{domain}_{src}-{trg}"
        print(f"Training model ({fname_base})...")

        # Create paths
        Path(os.path.join(dataset, DATASET_LOGS_NAME)).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(dataset, DATASET_CHECKPOINT_NAME)).mkdir(parents=True, exist_ok=True)

        # Train model
        run_experiment(dataset, src, trg, model_name=MODEL_NAME, domain=domain)


# # Get output
# outputs = model.translate_batch(src, src_key_padding_mask, max_length=100, beam_width=1)
#
# # Get translations
# trg_pred = [x[0][0] for x in outputs]  # Get best
# src_dec, ref_dec, hyp_dec = get_translations(src, trg, trg_pred, model.src_tok, model.trg_tok)
# src_dec_all += src_dec
# ref_dec_all += ref_dec
# hyp_dec_all += hyp_dec
# break

# # Print translations
# if print_translations:
#     helpers.print_translations(hypothesis=hyp_dec_all, references=ref_dec_all, source=src_dec_all, limit=None)
#
# # Compute metrics
# metrics = {}
# torch_bleu = torchtext.data.metrics.bleu_score([x.split(" ") for x in hyp_dec_all],
#                                                [[x.split(" ")] for x in ref_dec_all])
# metrics["torch_bleu"] = torch_bleu