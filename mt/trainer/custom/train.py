import os
import numpy as np
import random
import time
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from tqdm import tqdm

from mt.preprocess import utils
from mt import helpers
from mt import DATASETS_PATH, LOGS_PATH
from mt.trainer.models.pytransformer.transformer import TransformerModel


MODEL_NAME = "transformer"
BPE_FOLDER = "bpe.8000"

MAX_EPOCHS = 1000
LEARNING_RATE = 1e-2
DEVICE1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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


def init_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


def run_experiment(datapath, src, trg, model_name, bpe_folder, domain=None, batch_size=32//2, max_tokens=4096//2, num_workers=0):
    # Load tokenizers
    src_tok, trg_tok = helpers.get_tokenizers(os.path.join(datapath, bpe_folder), src, trg, use_fastbpe=True)  # use_fastbpe != apply_fastbpe

    # Load dataset
    datasets = helpers.load_dataset(os.path.join(datapath, bpe_folder), src, trg, splits=["train", "val", "test"])

    # Prepare data loaders
    train_loader = helpers.build_dataloader(datasets["val"], src_tok, trg_tok, batch_size=batch_size, max_tokens=max_tokens, num_workers=num_workers)
    val_loader = helpers.build_dataloader(datasets["val"], src_tok, trg_tok, batch_size=batch_size, max_tokens=max_tokens, num_workers=num_workers, shuffle=False)
    # test_loader = helpers.build_dataloader(datasets["test"], src_tok, trg_tok, batch_size=batch_size, max_tokens=max_tokens, num_workers=num_workers, shuffle=False)

    # Instantiate model #1
    model1 = TransformerModel(src_tok=src_tok, trg_tok=trg_tok)
    model1.apply(init_weights)
    model1.to(DEVICE1)
    optimizer1 = optim.Adam(model1.parameters(), lr=LEARNING_RATE)

    # Set loss (ignore when the target token is <pad>)
    criterion = nn.CrossEntropyLoss(ignore_index=trg_tok.word2idx[trg_tok.PAD_WORD])

    # Tensorboard (it needs some epochs to start working ~10-20)
    tr_writer = SummaryWriter(f"{model_name}/train")
    val_writer = SummaryWriter(f"{model_name}/val")

    # Train and validate model
    fit((model1, optimizer1), (None, None), train_loader, val_loader=val_loader,
        epochs=MAX_EPOCHS, criterion=criterion,
        checkpoint_path=os.path.join(datapath, "checkpoint.pt"), tr_writer=tr_writer, val_writer=val_writer)

    print("Done!")


def fit(model_opt1, model_opt2, train_loader, val_loader, epochs, criterion, checkpoint_path, tr_writer=None, val_writer=None):
    for epoch in range(epochs):
        start_time = time.time()
        n_iter = epoch + 1

        # Train model
        train(model_opt1, model_opt2, train_loader, criterion, epoch_i=n_iter, tb_writer=tr_writer)

        # # Evaluate model
        # evaluate(model, val_loader, criterion, epoch_i=n_iter, tb_writer=val_writer)


def train(model_opt1, model_opt2, train_loader, criterion, clip=0.25, log_interval=1, epoch_i=None, tb_writer=None):
    total_loss = 0.0
    start_time = time.time()

    # Unpack values
    (model1, optimizer1) = model_opt1
    (model2, optimizer2) = model_opt2

    model1.train()
    # for i, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
    for i, batch in enumerate(train_loader):
        # Get batch data
        src1, src_mask1, trg1, trg_mask1 = [x.to(DEVICE1) for x in batch]
        src_vocab1, trg_vocab1 = model1.src_tok.get_vocab_size(), model1.trg_tok.get_vocab_size()
        batch_size, src_max_len, trg_max_len = src1.shape[0], src1.shape[1], trg1.shape[1]

        # Zero the parameter gradients
        optimizer1.zero_grad()

        # Get output
        output1 = model1(src1, src_mask1, trg1[:, :-1], trg_mask1[:, :-1])
        _output1 = output1.detach().permute(1, 0, 2)
        _src1 = src1.detach()
        _trg1 = trg1.detach()

        # For debugging
        # We need to reshape the output to get the maximum probabilities for earch batch and position
        # We subtract 1 to the max_len due to the <sos> removal
        idxs = torch.argmax(output1.view(batch_size, trg_max_len - 1, trg_vocab1), dim=2)

        src_dec = model1.src_tok.decode(_src1)
        hyp_dec = model1.trg_tok.decode(idxs.view(batch_size, -1))
        ref_dec = model1.trg_tok.decode(_trg1.detach().view(batch_size, -1))
        helpers.print_translations(hypothesis=hyp_dec, references=ref_dec, source=src_dec, limit=1)

        # Reshape output / target
        # Let's assume that after the <eos> everything has be predicted as <pad>,
        # and then, we will ignore the pads in the CrossEntropy
        output1 = output1.contiguous().view(-1, trg_vocab1)  # (B, L, vocab) => (B*L, vocab)
        trg1 = trg1[:, 1:].contiguous().view(-1)  # Remove <sos> and reshape to vector (B*L)

        # Compute loss and backward => CE(I(N, C), T(N))
        loss = criterion(output1, trg1)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        #torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        total_loss += loss.item()
        optimizer1.step()

        # Log progress
        if (i+1) % log_interval == 0:
            log_progress("train", total_loss, epoch_i+1, i+1, len(train_loader), start_time, tb_writer)


def log_progress(prefix, total_loss, epoch_i, batch_i, n_batches, start_time, tb_writer):
    elapsed = time.time() - start_time
    total_minibatches = (epoch_i - 1) * n_batches + batch_i

    # Compute metrics
    cur_loss = total_loss / batch_i
    ppl = math.exp(cur_loss)

    # Print stuff
    print('| epoch: {:<3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | loss {:5.2f} | ppl {:8.2f}'.format(
        epoch_i, batch_i, n_batches, (elapsed * 1000) / batch_i, cur_loss, ppl))

    # Tensorboard
    if tb_writer:
        tb_writer.add_scalar(f'{prefix}_Loss/batch', cur_loss, total_minibatches)
        tb_writer.add_scalar(f'{prefix}_PPL/batch', ppl, total_minibatches)


def evaluate(model, test_iter, criterion, epoch_i=None, tb_writer=None, tb_batch_rate=None):
    model.eval()
    epoch_loss = 0

    # with torch.no_grad():
    #     for i, batch in tqdm(enumerate(test_iter), total=len(test_iter)):
    #
    #
    #         ##############################
    #
    #         ##############################
    #
    #         # Compute loss and backward => CE(I(N, C), T(N))
    #         loss = criterion(output, trg)
    #
    #         epoch_loss += loss.item()
    #
    #         # Tensorboard
    #         if tb_writer and i % tb_batch_rate == 0:
    #             bn_iter = (n_iter - 1) * len(test_iter) + (i + 1)
    #             b_loss = epoch_loss / (i + 1)
    #             tb_writer.add_scalar('Loss/batch', b_loss, bn_iter)
    #             tb_writer.add_scalar('PPL/batch', math.exp(b_loss), bn_iter)

    return epoch_loss / len(test_iter)


def summary_report(train_loss=None, test_loss=None, start_time=None, tr_writer=None, val_writer=None, n_iter=0, testing=False):
    # Print summary
    if start_time:
        end_time = time.time()
        epoch_mins, epoch_secs = utils.epoch_time(start_time, end_time)
        print(f'Epoch: {n_iter:02} | Time: {epoch_mins}m {epoch_secs}s')
    else:
        print(f"Summary report:")

    # Metrics
    if train_loss is not None:
        # Metrics
        train_ppl = math.exp(train_loss)

        # Tensorboard
        if tr_writer:
            tr_writer.add_scalar('Loss', train_loss, n_iter)
            tr_writer.add_scalar('PPL', train_ppl, n_iter)

    # Validation
    if test_loss is not None:
        test_type = "Test" if testing else "Val."

        # Metrics
        test_ppl = math.exp(test_loss)
        print(f'\t {test_type} Loss: {test_loss:.3f} |  {test_type} PPL: {test_ppl:7.3f}')

        # Tensorboard
        if val_writer:
            val_writer.add_scalar('Loss', test_loss, n_iter)
            val_writer.add_scalar('PPL', test_ppl, n_iter)


if __name__ == "__main__":
    # Get all folders in the root path
    # datasets = [os.path.join(DATASETS_PATH, name) for name in os.listdir(DATASETS_PATH) if os.path.isdir(os.path.join(DATASETS_PATH, name))]
    datasets = [os.path.join(DATASETS_PATH, "tmp|health_es-en")]
    for dataset in datasets:
        domain, (src, trg) = utils.get_dataset_ids(dataset)
        fname_base = f"{domain}_{src}-{trg}"
        print(f"Training model ({fname_base})...")

        # Train model
        run_experiment(dataset, src, trg, model_name=MODEL_NAME, bpe_folder=BPE_FOLDER, domain=domain)
