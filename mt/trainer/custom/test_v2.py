import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext
from torch.utils.data import DataLoader
from tqdm import tqdm

from mt import DATASETS_PATH, DATASET_CLEAN_NAME, DATASET_TOK_NAME, DATASET_LOGS_NAME, DATASET_CHECKPOINT_NAME
from mt import helpers
from mt.preprocess import utils
from mt.trainer.datasets import TranslationDataset
from mt.trainer.models.transformer.transformer import Transformer

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
    # Load tokenizers
    src_tok, trg_tok = helpers.get_tokenizers(os.path.join(datapath, DATASET_TOK_NAME, TOK_FOLDER), src, trg, tok_model=TOK_MODEL, lower=LOWERCASE)

    # Load dataset
    test_ds = TranslationDataset(os.path.join(datapath, DATASET_CLEAN_NAME), src_tok, trg_tok, "test")
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=lambda x: TranslationDataset.collate_fn(x, MAX_TOKENS), pin_memory=True)

    # Instantiate model #1
    model = Transformer(d_model=256,
                        enc_layers=3, dec_layers=3,
                        enc_heads=8, dec_heads=8,
                        enc_dff_dim=512, dec_dff_dim=512,
                        enc_dropout=0.1, dec_dropout=0.1,
                        max_src_len=200, max_trg_len=200,
                        src_tok=src_tok, trg_tok=trg_tok,
                        static_pos_emb=False).to(DEVICE1)
    print(f'The model has {model.count_parameters():,} trainable parameters')
    criterion = nn.CrossEntropyLoss(ignore_index=trg_tok.word2idx[trg_tok.PAD_WORD])

    # Load weights
    checkpoint_path = os.path.join(datapath, DATASET_CHECKPOINT_NAME, "transformer_multi30k_best.pt")
    print(f"Loading weights from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path))

    # # Evaluate
    # start_time = time.time()
    # val_loss, translations = evaluate(model, test_loader, criterion)
    #
    # # Log progress
    # metrics = log_progress(start_time, val_loss, translations)

    # Get bleu
    src_dec_all, hyp_dec_all, ref_dec_all = get_translations(test_loader, model)

    bleu_score = torchtext.data.metrics.bleu_score([x.split() for x in hyp_dec_all], [[x.split()] for x in ref_dec_all])
    print(f'BLEU score = {bleu_score * 100:.2f}')

    #
    # # Get translations
    # src_dec_all, hyp_dec_all, ref_dec_all = get_translations(model, test_loader)
    #
    # # Print translations
    # helpers.print_translations(hyp_dec_all, ref_dec_all, src_dec_all, limit=50)
    #
    # # Compute scores
    # m_bleu_score = bleu_score([x.split(" ") for x in hyp_dec_all], [[x.split(" ")] for x in ref_dec_all])
    # print(f'BLEU score = {m_bleu_score*100:.3f}')
    #
    # # Create path
    # eval_name = domain
    # eval_path = os.path.join(datapath, DATASET_EVAL_NAME, eval_name)
    # Path(eval_path).mkdir(parents=True, exist_ok=True)
    #
    # # Save translations to file
    # with open(os.path.join(eval_path, 'src.txt'), 'w') as f:
    #     f.writelines("%s\n" % s for s in src_dec_all)
    # with open(os.path.join(eval_path, 'hyp.txt'), 'w') as f:
    #     f.writelines("%s\n" % s for s in hyp_dec_all)
    # with open(os.path.join(eval_path, 'ref.txt'), 'w') as f:
    #     f.writelines("%s\n" % s for s in ref_dec_all)
    # print("Translations written!")



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
        m_bleu_score = torchtext.data.metrics.bleu_score([x.split(" ") for x in hyp_dec_all], [[x.split(" ")] for x in ref_dec_all])
        metrics["val"]["bleu"] = m_bleu_score*100

        # Print translations
        helpers.print_translations(hyp_dec_all, ref_dec_all, src_dec_all, limit=50)

    # Print stuff
    end_time = time.time()
    epoch_mins, epoch_secs = helpers.epoch_time(start_time, end_time)
    print("------------------------------------------------------------")
    print(f'Evaluate | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\t- Val Loss: {metrics["val"]["loss"]:.3f} | Val PPL: {metrics["val"]["ppl"]:.3f} | Val BLEU: {metrics["val"]["bleu"]:.3f}')
    print("------------------------------------------------------------")

    return metrics


def translate_sentence(src, model, max_length=50, beam_width=3):
    model.eval()

    batch_size = src.shape[0]
    src_tensor = src.long().to(DEVICE1)
    src_mask = model.make_src_mask2(src_tensor)

    with torch.no_grad():
        src_tensor = model.enc_input(src_tensor)
        enc_src = model.encoder(src_tensor, src_mask)

    # Prepare target inputs (sos)
    TRG_SOS_IDX = model.trg_tok.word2idx[model.trg_tok.SOS_WORD]
    TRG_EOS_IDX = model.trg_tok.word2idx[model.trg_tok.EOS_WORD]
    trg = torch.LongTensor([[TRG_SOS_IDX] for _ in range(len(src_tensor))]).to(enc_src.device)
    beam_probs = torch.ones((len(src_tensor), beam_width)).to(enc_src.device)

    for i in range(max_length):
        b_probs = []
        b_idxs = []

        for b in range(beam_width):
            # Encode target tensor
            _trg = trg[:, b, :] if trg.ndim > 2 else trg
            trg_mask = model.make_trg_mask2(_trg)
            with torch.no_grad():
                trg_tensor = model.dec_input(_trg)
                output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
                output = model.fc_out(output)  # (B, L, d_model) => (B, L, vocab)
                output = F.softmax(output, dim=2)  # (B, L, d_model) => (B, L, vocab)

            #next_tokens = output.argmax(2)[:, -1].unsqueeze(1)
            # Get top-beam candidates per batch
            probs, idxs = output[:, -1].sort(dim=1, descending=True)
            probs = probs[:, :beam_width]  # .permute(0, 2, 1)  # Minor optimization
            idxs = idxs[:, :beam_width]  # .permute(0, 2, 1)
            b_probs.append(probs)
            b_idxs.append(idxs)

            if trg.ndim == 2:
                break

        # P(old_beams) * P(new_token)
        # Merge all beams
        new_idxs = []
        for bi, (probs, idxs) in enumerate(zip(b_probs, b_idxs)):  # first len(bi)==1
            prob = beam_probs[:, bi].unsqueeze(1) * probs  # / len  Normalize
            _trg = trg[:, bi, :].unsqueeze(1) if trg.ndim > 2 else trg.unsqueeze(1)
            idxs = torch.cat([_trg.repeat(1, beam_width, 1), idxs.unsqueeze(2)], dim=2)
            new_idxs.append([idxs, prob])

        # Gest top-k: [a, b, c], [d, e, f], [g, h, i] => [b,f, i]
        idxs = torch.cat([x[0] for x in new_idxs], dim=1)
        probs = torch.cat([x[1] for x in new_idxs], dim=1)
        beam_probs, probidxs = torch.topk(probs, beam_width)
        trg = torch.cat([x[y].unsqueeze(0) for x, y in zip(idxs, probidxs)])
        asdas = 2

    # Split sentences by beam search
    trg = [trg[:, i, :] for i in range(beam_width)]
    return trg, attention

def translate_batch(model, src, src_mask, max_length=150, beam_width=3):
    # Build source mask
    src_tensor = src.long().to(DEVICE1)
    src_mask = model.make_src_mask2(src_tensor)

    with torch.no_grad():
        src_tensor = model.enc_input(src_tensor)
        enc_src = model.encoder(src_tensor, src_mask)

    # Prepare target inputs (sos)
    TRG_SOS_IDX = model.trg_tok.word2idx[model.trg_tok.SOS_WORD]
    TRG_EOS_IDX = model.trg_tok.word2idx[model.trg_tok.EOS_WORD]
    trg = torch.LongTensor([[TRG_SOS_IDX] for _ in range(len(src))]).to(enc_src.device)

    for i in range(max_length):
        b_probs = []
        b_idxs = []
        for b in range(beam_width):
            # Encode target tensor
            _trg = trg[:, b, :] if trg.ndim > 2 else trg
            trg_mask = model.make_trg_mask2(_trg)

            with torch.no_grad():
                enc_trg = model.dec_input(_trg)
                output, attention = model.decoder(enc_trg, enc_src, trg_mask, src_mask)
                output = model.fc_out(output)  # (B, L, d_model) => (B, L, vocab)
                output = F.log_softmax(output)  # (B, L, d_model) => (B, L, vocab)

            # Get top-k candidates per batch
            probs, idxs = output[:, -1].sort(dim=1, descending=True)
            probs = probs[:, :beam_width]  # .permute(0, 2, 1)  # Minor optimization
            idxs = idxs[:, :beam_width]  # .permute(0, 2, 1)
            b_probs.append(probs)
            b_idxs.append(idxs)

            if trg.ndim == 2:
                break

        # Merge all beams
        b_idxs = torch.cat(b_idxs, dim=1)  # .squeeze(2)
        b_probs = torch.cat(b_probs, dim=1)  # .squeeze(2)

        # Sort candidates
        probs, probsidxs_ = b_probs.sort(dim=1, descending=True)
        probs = probs[..., :beam_width]
        probsidxs_ = probsidxs_[..., :beam_width]
        idxs_ = torch.cat([b_idxs[i, probsidxs_[i]].unsqueeze(0) for i in range(len(b_idxs))])

        if trg.ndim == 2:
            trg = trg.repeat(1, beam_width).unsqueeze(2)
        trg = torch.cat([trg, idxs_.unsqueeze(2)], dim=2)
        sdasd = 3

    # Split sentences by beam search
    pred_sentences = [trg[:, i, :] for i in range(beam_width)]
    return pred_sentences


def get_translations(data, model, max_len=50, beam_width=3):
    src_dec_all, hyp_dec_all, ref_dec_all = [], [], []

    for batch in tqdm(data, total=len(data)):
        # Get batch data
        src, src_mask, trg, trg_mask = [x.to(DEVICE1) for x in batch]
        batch_size, src_max_len, trg_max_len = src.shape[0], src.shape[1], trg.shape[1]

        pred_trg, _ = translate_sentence(src, model, max_len)
        # pred_trg = translate_batch(model, src, src_mask, max_length=max_len, beam_width=beam_width)
        pred_trg = pred_trg[0]

        # cut off <eos> token
        hyp_dec_all += model.trg_tok.decode(pred_trg, remove_special_tokens=True)
        ref_dec_all += model.trg_tok.decode(trg, remove_special_tokens=True)
        src_dec_all += model.src_tok.decode(src, remove_special_tokens=True)

    return src_dec_all, hyp_dec_all, ref_dec_all


if __name__ == "__main__":
    # Get all folders in the root path
    # datasets = [os.path.join(DATASETS_PATH, x) for x in ["health_es-en", "biological_es-en", "merged_es-en"]]
    datasets = [os.path.join(DATASETS_PATH, "multi30k_de-en")]
    for dataset in datasets:
        domain, (src, trg) = utils.get_dataset_ids(dataset)
        fname_base = f"{domain}_{src}-{trg}"
        print(f"Training model ({fname_base})...")

        # Create paths
        Path(os.path.join(dataset, DATASET_LOGS_NAME)).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(dataset, DATASET_CHECKPOINT_NAME)).mkdir(parents=True, exist_ok=True)

        # Train model
        run_experiment(dataset, src, trg, model_name=MODEL_NAME, domain=domain)
