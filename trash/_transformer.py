import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x.unsqueeze(2)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Embedding(nn.Module):

    def __init__(self, vocab_size, d_model, dropout, max_length):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)  # Vocab => emb
        self.pos_emb = PositionalEncoding(d_model, dropout, max_length)  # Pos => emb_pos
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.nn.Parameter(torch.sqrt(torch.FloatTensor([d_model])), requires_grad=False)

    def forward(self, src):
        # Mix token embeddings and positional embeddings
        src = self.dropout((self.embedding(src) * self.scale) + self.pos_emb(src))  # (B, src_len, hid_dim)
        return src


class LitTransformer(pl.LightningModule):

    def __init__(self, src_vocab, trg_vocab, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=2048, dropout=0.1, max_length=5000):
        super().__init__()

        self.enc_emb = Embedding(vocab_size=src_vocab, d_model=d_model, dropout=dropout, max_length=max_length)
        self.dec_emb = Embedding(vocab_size=trg_vocab, d_model=d_model, dropout=dropout, max_length=max_length)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead,
                                          num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward, dropout=dropout)

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        pass

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        src, src_mask, trg, trg_mask = batch
        assert len(src) == len(src_mask)
        assert len(trg) == len(trg_mask)

        # Predict
        x_src = self.enc_emb(src)
        x_trg = self.dec_emb(trg)
        output = self.model(x_src, x_trg, src_mask, trg_mask)

        # Logging to TensorBoard by default
        loss = F.log_softmax(output, dim=-1)
        self.log('train_loss', loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


