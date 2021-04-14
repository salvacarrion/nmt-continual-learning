import os
import math
import time
import numpy as np
import random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from mt.preprocess import utils
from mt import helpers
from mt.trainer.models.transformer_old.transformer import Encoder, Decoder, Seq2Seq, init_weights
from mt.helpers import print_translations


MODEL_NAME = "transformer"
BPE_FOLDER = "bpe.8000"

# Deterministic environment
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Use zero workers when debugging to avoid freezing
def train_model(datapath, src, trg, model_name, bpe_folder, domain=None, batch_size=32, max_tokens=4096, num_workers=0):
    # Load tokenizers
    lt_src, lt_trg = helpers.get_tokenizers(os.path.join(datapath, bpe_folder), src, trg, use_fastbpe=True)  # use_fastbpe != apply_fastbpe

    # Load dataset
    datasets = helpers.load_dataset(os.path.join(datapath, bpe_folder), src, trg, splits=["train", "val", "test"])

    # Prepare data loaders
    train_loader = helpers.build_dataloader(datasets["val"], lt_src, lt_trg, batch_size=batch_size, max_tokens=max_tokens, num_workers=num_workers)
    val_loader = helpers.build_dataloader(datasets["val"], lt_src, lt_trg, batch_size=batch_size, max_tokens=max_tokens, num_workers=num_workers, shuffle=False)
    # test_loader = helpers.build_dataloader(datasets["test"], lt_src, lt_trg, batch_size=batch_size, max_tokens=max_tokens, num_workers=num_workers, shuffle=False)

    # Instantiate model
    hid_dim = 256
    enc_layers = 3
    dec_layers = 3
    enc_heads = 8
    dec_heads = 8
    enc_pf_dim = 512
    dec_pf_dim = 512
    enc_dropout = 0.1
    dec_dropout = 0.1
    src_vocab, trg_vocab = lt_src.get_vocab_size(), lt_trg.get_vocab_size()
    enc = Encoder(src_vocab, hid_dim, enc_layers, enc_heads, enc_pf_dim, enc_dropout, lt_src.max_length)
    dec = Decoder(trg_vocab, hid_dim, dec_layers, dec_heads, dec_pf_dim, dec_dropout, lt_trg.max_length)
    model = Seq2Seq(enc, dec, lt_src, lt_trg)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Initialize model
    model.apply(init_weights)

    # Set loss (ignore when the target token is <pad>)
    pad_idx = lt_trg.word2idx[lt_trg.PAD_WORD]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    # Set optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Fit model
    for i in range(1, 50+1):
        train_loss = train(model, train_loader, optimizer, criterion, device, clip=1.0, n_iter=i, src_tok=lt_src, trg_tok=lt_trg)
        print(f"Train loss: {train_loss}")


def train(model, train_iter, optimizer, criterion, device, clip, n_iter=None, src_tok=None, trg_tok=None, show_translations=True):
    model.train()
    epoch_loss = 0

    for i, batch in tqdm(enumerate(train_iter), total=len(train_iter)):
        # Get data
        src, src_mask, trg, trg_mask = batch
        src, src_mask, trg, trg_mask = src.to(device), src_mask.to(device), trg.to(device), trg_mask.to(device)
        batch_size = src.shape[0]
        src_vocab, trg_vocab = src_tok.get_vocab_size(), trg_tok.get_vocab_size()
        src_max_len, trg_max_len = src.shape[1], trg.shape[1]

        # Reset grads and get output
        optimizer.zero_grad()

        ##############################
        # Feed input
        output, _ = model(src, trg[:, :-1])  # Ignore <eos> as input for trg

        # Reshape output / target
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)  # (B, L, vocab) => (B*L, vocab)
        trg = trg[:, 1:].contiguous().view(-1)  # Remove <sos> and reshape to vector (B*L)
        ##############################

        # Compute loss and backward => CE(I(N, C), T(N))
        loss = criterion(output, trg)
        loss.backward()

        # Clip grads and update parameters
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
        print(f"=> Epoch: #{n_iter} | Minibatch #{i+1}: loss={epoch_loss / (i+1)}")

        # For debugging
        if show_translations:
            with torch.no_grad():
                src_dec = src_tok.decode(src)
                hyp_dec = trg_tok.decode(torch.argmax(output.detach().reshape(batch_size, trg_max_len-1, trg_vocab), dim=2).reshape(batch_size, -1))
                ref_dec = trg_tok.decode(trg.detach().reshape(batch_size, -1))
                print_translations(hypothesis=hyp_dec, references=ref_dec, source=src_dec, limit=1)

        # # Tensorboard
        # if tb_writer and i % tb_batch_rate == 0:
        #     bn_iter = (n_iter-1) * len(train_iter) + (i+1)
        #     b_loss = epoch_loss / (i+1)
        #     tb_writer.add_scalar('Loss/batch', b_loss, bn_iter)
        #     tb_writer.add_scalar('PPL/batch', math.exp(b_loss), bn_iter)

    return epoch_loss / len(train_iter)


if __name__ == "__main__":
    # Get all folders in the root path
    # datasets = [os.path.join(DATASETS_PATH, name) for name in os.listdir(DATASETS_PATH) if os.path.isdir(os.path.join(DATASETS_PATH, name))]
    datasets = [os.path.join(DATASETS_PATH, "tmp|health_es-en")]
    for dataset in datasets:
        domain, (src, trg) = utils.get_dataset_ids(dataset)
        fname_base = f"{domain}_{src}-{trg}"
        print(f"Training model ({fname_base})...")

        # Train model
        train_model(dataset, src, trg, model_name=MODEL_NAME, bpe_folder=BPE_FOLDER, domain=domain)
