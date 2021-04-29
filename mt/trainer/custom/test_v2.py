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
import spacy
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
from mt import DATASETS_PATH, DATASET_EVAL_NAME, DATASET_CLEAN_NAME, DATASET_TOK_NAME, DATASET_LOGS_NAME, DATASET_CHECKPOINT_NAME
from mt.trainer.models.pytransformer.transformer import TransformerModel
from mt.trainer.models.optim import  ScheduledOptim
from mt.trainer.models.pytransformer.transformer_bv import Encoder, Decoder, Seq2Seq
from mt.trainer.tok import word_tokenizer

MODEL_NAME = "transformer_bv"


MAX_EPOCHS = 50
LEARNING_RATE = 0.0005 #1e-3
BATCH_SIZE = 32 #int(32*1.5)
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
    test_ds = TranslationDataset(os.path.join(datapath, DATASET_CLEAN_NAME), src_tok, trg_tok, "test")
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=TranslationDataset.collate_fn, pin_memory=True)

    # # Instantiate model #1
    # model1 = TransformerModel(src_tok=src_tok, trg_tok=trg_tok)
    # # model1.load_state_dict(torch.load(checkpoint_path))
    # optimizer1 = ScheduledOptim(
    #     optim.Adam(model1.parameters(), betas=(0.9, 0.98), eps=1e-09, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY),
    #     model1.d_model, WARMUP_UPDATES)
    # if MULTIGPU and torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model1 = nn.DataParallel(model1)
    # model1.to(DEVICE1)
    # # wandb.watch(model1)
    #
    # # Set loss (ignore when the target token is <pad>)
    # criterion = nn.CrossEntropyLoss(ignore_index=trg_tok.word2idx[trg_tok.PAD_WORD])

    INPUT_DIM = src_tok.get_vocab_size()
    OUTPUT_DIM = trg_tok.get_vocab_size()
    HID_DIM = 256
    ENC_LAYERS = 3
    DEC_LAYERS = 3
    ENC_HEADS = 8
    DEC_HEADS = 8
    ENC_PF_DIM = 512
    DEC_PF_DIM = 512
    ENC_DROPOUT = 0.1
    DEC_DROPOUT = 0.1
    enc = Encoder(INPUT_DIM,
                  HID_DIM,
                  ENC_LAYERS,
                  ENC_HEADS,
                  ENC_PF_DIM,
                  ENC_DROPOUT,
                  DEVICE1)
    dec = Decoder(OUTPUT_DIM,
                  HID_DIM,
                  DEC_LAYERS,
                  DEC_HEADS,
                  DEC_PF_DIM,
                  DEC_DROPOUT,
                  DEVICE1)
    SRC_PAD_IDX = src_tok.word2idx[src_tok.PAD_WORD]
    TRG_PAD_IDX = trg_tok.word2idx[trg_tok.PAD_WORD]
    model1 = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, DEVICE1).to(DEVICE1)
    model1.apply(initialize_weights)
    print(f'The model has {count_parameters(model1):,} trainable parameters')
    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

    model1.load_state_dict(torch.load(checkpoint_path))

    # test_loss, _ = evaluate(model1, test_loader, criterion)
    # print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

    src_dec_all, hyp_dec_all, ref_dec_all = get_translations(model1, test_loader, src_tok, trg_tok)

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

    m_bleu_score = bleu_score([x.split(" ") for x in hyp_dec_all], [[x.split(" ")] for x in ref_dec_all])
    print(f'BLEU score = {m_bleu_score*100:.2f}')


def evaluate(model, data_loader, criterion):
    src_dec_all, ref_dec_all, hyp_dec_all = [], [], []
    epoch_loss = 0
    metrics = {}

    model.eval()
    for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
        with torch.no_grad():
            # print(i)

            # Get batch data
            src1, src_mask1, trg1, trg_mask1 = [x.to(DEVICE1) for x in batch]
            batch_size, src_max_len, trg_max_len = src1.shape[0], src1.shape[1], trg1.shape[1]

            output, _ = model(src1, trg1[:, :-1])

            # output = [batch size, trg len - 1, output dim]
            # trg = [batch size, trg len]

            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            trg1 = trg1[:, 1:].contiguous().view(-1)

            # output = [batch size * trg len - 1, output dim]
            # trg = [batch size * trg len - 1]

            loss = criterion(output, trg1.long())

            epoch_loss += loss.item()

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

    return epoch_loss / len(data_loader), (src_dec_all, ref_dec_all, hyp_dec_all)


def get_translations(model, data_loader, src_tok, trg_tok, max_len=100):
    srcs = []
    trgs = []
    pred_trgs = []

    model.eval()
    for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
        # Get batch data
        src1, src_mask1, trg1, trg_mask1 = [x.to(DEVICE1) for x in batch]
        batch_size, src_max_len, trg_max_len = src1.shape[0], src1.shape[1], trg1.shape[1]

        pred_trg, _ = translate_sentence_vectorized(src1, src_tok, trg_tok, model, max_len)

        pred_trgs += trg_tok.decode(pred_trg)
        trgs += trg_tok.decode(trg1)
        srcs += src_tok.decode(src1)
        break

    return srcs, pred_trgs, trgs


def translate_sentence_vectorized(src_tensor, src_tok, trg_tok, model, max_len):
    assert isinstance(src_tensor, torch.Tensor)

    model.eval()
    src_mask = model.make_src_mask(src_tensor)

    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)
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
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
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

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


if __name__ == "__main__":
    # Get all folders in the root path
    #datasets = [os.path.join(DATASETS_PATH, name) for name in os.listdir(DATASETS_PATH) if os.path.isdir(os.path.join(DATASETS_PATH, name))]
    # datasets = [os.path.join(DATASETS_PATH, "multi30k_de-en")]
    datasets = [os.path.join(DATASETS_PATH, "health_es-en")]
    for dataset in datasets:
        domain, (src, trg) = utils.get_dataset_ids(dataset)
        fname_base = f"{domain}_{src}-{trg}"
        print(f"Training model ({fname_base})...")

        # Create paths
        Path(os.path.join(dataset, DATASET_LOGS_NAME)).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(dataset, DATASET_CHECKPOINT_NAME)).mkdir(parents=True, exist_ok=True)

        # Train model
        run_experiment(dataset, src, trg, model_name=MODEL_NAME, domain=domain)
