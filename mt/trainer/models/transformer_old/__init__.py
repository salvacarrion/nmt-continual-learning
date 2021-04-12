import os
from collections import defaultdict
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.utils.data import random_split

from tokenizers import Tokenizer, normalizers, pre_tokenizers, decoders
from tokenizers.models import WordPiece
from tokenizers.normalizers import NFD, Lowercase, Strip, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordPieceTrainer
from tokenizers.processors import TemplateProcessing


class LitTransfomer(pl.LightningModule):

    def __init__(self, tokenizer,
                 d_model=512,
                 enc_layers=2, dec_layers=2,
                 enc_heads=8, dec_heads=8,
                 enc_dff_dim=2048, dec_dff_dim=2048,
                 enc_dropout=0.1, dec_dropout=0.1,
                 max_src_len=2000, max_trg_len=2000):
        super().__init__()

        # Some variables
        self.batch_size = 32
        self.learning_rate = 10e-3

        # Set tokenizer
        self.tokenizer = tokenizer

        # Vocab sizes
        input_dim = self.tokenizer.src_tokenizer.get_vocab_size()
        output_dim = self.tokenizer.trg_tokenizer.get_vocab_size()

        # Model
        self.enc = tfmr.Encoder(input_dim, d_model, enc_layers, enc_heads, enc_dff_dim, enc_dropout, max_src_len)
        self.dec = tfmr.Decoder(output_dim, d_model, dec_layers, dec_heads, dec_dff_dim, dec_dropout, max_trg_len)
        self.model = tfmr.Seq2Seq(self.enc, self.dec)

        # Initialize weights
        self.model.apply(tfmr.init_weights)

        # Set loss (ignore when the target token is <pad>)
        TRG_PAD_IDX = self.tokenizer.trg_tokenizer.token_to_id(self.tokenizer.PAD_WORD)
        self.criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

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
        output, _ = self.model(src, src_mask, trg[:, :-1], trg_mask[:, :-1])

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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
