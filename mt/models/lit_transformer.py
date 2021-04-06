import os
from collections import defaultdict
import math

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import sacrebleu

from mt.models.transformer import Transformer


def init_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


class LitTransformer(pl.LightningModule):

    def __init__(self, lt_src, lt_trg):
        super().__init__()

        # Save tokenizers
        self.src_tok = lt_src
        self.trg_tok = lt_trg

        # Model params
        self.transformer = Transformer(self.src_tok.get_vocab_size(), self.trg_tok.get_vocab_size())

        # Initialize weights
        self.transformer.apply(init_weights)

        # Set loss (ignore when the target token is <pad>)
        pad_idx = self.trg_tok.word2idx[lt_trg.PAD_WORD]
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    def forward(self, x):
        # Inference
        return x

    def training_step(self, batch, batch_idx):
        # Run one mini-batch
        _, losses, metrics = self._batch_step(batch, batch_idx)

        # Logging to TensorBoard by default
        self.log('train_loss', losses['loss'])
        self.log('train_ppl', metrics['ppl'])
        return losses['loss']

    def validation_step(self, batch, batch_idx):
        loss = self._shared_eval(batch, batch_idx, 'val')
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._shared_eval(batch, batch_idx, 'test')
        return loss

    def _batch_step(self, batch, batch_idx):
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
        _output = output.contiguous().view(-1, output_dim)  # (B, L, vocab) => (B*L, vocab)
        _trg = trg[:, 1:].contiguous().view(-1)  # Remove <sos> and reshape to vector (B*L)
        ##############################

        # Compute loss and metrics
        losses = {'loss': self.criterion(_output, _trg)}
        metrics = {'ppl': math.exp(losses['loss'])}
        return output, losses, metrics

    def _shared_eval(self, batch, batch_idx, prefix):
        # Run one mini-batch
        output, losses, metrics = self._batch_step(batch, batch_idx)
        # src, src_mask, trg, trg_mask = batch

        # output_trg = torch.argmax(output, 2)
        # output_words = self.trg_tok.decode(output_trg)

        # Logging to TensorBoard by default
        self.log(f'{prefix}_loss', losses['loss'])
        self.log(f'{prefix}_ppl', metrics['ppl'])
        return losses['loss']

    def configure_optimizers(self, lr=1e-4):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        return optimizer
