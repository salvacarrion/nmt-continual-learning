import os
from collections import defaultdict
import math

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

from mt.models.transformer import Transformer


def init_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


class LitTransformer(pl.LightningModule):

    def __init__(self, src_vocab_size, trg_vocab_size, pad_idx):
        super().__init__()

        # Model params
        self.transformer = Transformer(src_vocab_size, trg_vocab_size)

        # Initialize weights
        self.transformer.apply(init_weights)

        # Set loss (ignore when the target token is <pad>)
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    def forward(self, x):
        # Inference
        return x

    def batch_step(self, batch, batch_idx):
        src, src_mask, trg, trg_mask = batch
        # trg_lengths = trg_mask.sum(dim=1) + 1  # Not needed

        # Feed input
        # src => whole sentence (including <sos>, <eos>)
        # src_mask => whole sentence (including <sos>, <eos>)
        # trg => whole sentence (except <eos> or use mask to remove it)
        # trg_mask => whole sentence except <eos>
        # NOTE: I remove the last token from TRG so that the prediction is L-1. This is
        # because later we will remove the <sos> from the TRG
        output, _ = self.transformer(src, src_mask, trg[:, :-1], trg_mask[:, :-1])

        # Reshape output / target
        # Let's presume that after the <eos> everything has be predicted as <pad>,
        # and then, we will ignore the pads in the CrossEntropy
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)  # (B, L, vocab) => (B*L, vocab)
        trg = trg[:, 1:].contiguous().view(-1)  # Remove <sos> and reshape to vector (B*L)
        ##############################

        # Compute loss and metrics
        losses = {'loss': self.criterion(output, trg)}
        metrics = {'ppl': math.exp(losses['loss'])}
        return losses, metrics

    def training_step(self, batch, batch_idx):
        # Run one mini-batch
        losses, metrics = self.batch_step(batch, batch_idx)

        # Logging to TensorBoard by default
        self.log('train_loss', losses['loss'])
        self.log('train_ppl', metrics['ppl'])
        return losses['loss']

    def validation_step(self, batch, batch_idx):
        # Run one mini-batch
        losses, metrics = self.batch_step(batch, batch_idx)

        # Logging to TensorBoard by default
        self.log('val_loss', losses['loss'])
        self.log('val_ppl', metrics['ppl'])
        return losses['loss']

    def configure_optimizers(self, lr=1e-4):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        return optimizer
