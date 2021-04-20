import math
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder
import torch.nn.functional as F

from mt import helpers


class TransformerModel(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 src_tok=None, trg_tok=None):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.d_model = d_model

        # Enc emb
        self.src_tok = src_tok
        self.src_pos_enc = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=self.src_tok.max_length)
        self.src_tok_embedding = nn.Embedding(self.src_tok.get_vocab_size(), d_model)

        # Dec emb
        self.trg_tok = trg_tok
        self.trg_pos_enc = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=self.trg_tok.max_length)
        self.trg_tok_embedding = nn.Embedding(self.trg_tok.get_vocab_size(), d_model)

        # Core
        self.transformer_model = nn.Transformer(d_model=d_model, nhead=nhead,
                                                num_encoder_layers=num_encoder_layers,
                                                num_decoder_layers=num_decoder_layers,
                                                dim_feedforward=dim_feedforward, dropout=dropout)
        self.trg_decoder = nn.Linear(d_model, self.trg_tok.get_vocab_size())

    def forward(self, src, src_mask, trg, trg_mask):
        # # For debugging
        # source = self.src_tok.decode(src)
        # reference = self.trg_tok.decode(trg)
        # helpers.print_translations(source, reference)

        # Process source
        src = self.src_tok_embedding(src) * math.sqrt(self.d_model)
        src = self.src_pos_enc(src)
        # src_mask = self._make_src_mask(src_mask)
        src_mask = ~src_mask.type(torch.bool)

        # Process target
        trg = self.trg_tok_embedding(trg) * math.sqrt(self.d_model)
        trg = self.trg_pos_enc(trg)
        # trg_mask = self._make_trg_mask(trg_mask)
        trg_mask = ~trg_mask.type(torch.bool)

        # (B, L, E) => (L, B, E)
        src = src.permute(1, 0, 2)
        trg = trg.permute(1, 0, 2)

        # Get output
        output = self.transformer_model(src, trg, src_key_padding_mask=src_mask, tgt_key_padding_mask=trg_mask)
        output = self.trg_decoder(output)
        output = F.softmax(output, dim=-1)
        return output

    def _make_src_mask(self, src_mask):
        # Extend dimensions
        src_mask = src_mask.unsqueeze(1).unsqueeze(2)  # (B, n_heads=1, seq_len=1, seq_len)
        return src_mask

    def _make_trg_mask(self, trg_mask):
        # Extend dimensions
        trg_mask = trg_mask.unsqueeze(1).unsqueeze(2)  # (B, n_heads=1, seq_len=1, seq_len)

        # Diagonal matrix to hide next token (LxL)
        trg_len = trg_mask.shape[3]  # target (max) length
        trg_tri_mask = torch.tril(torch.ones((trg_len, trg_len), device=trg_mask.device)).bool()

        # Add pads to the diagonal matrix (LxL)&Pad
        # This is automatically broadcast (B, 1, 1, L) & (L, L) => (B, 1, L, L)
        trg_mask = trg_mask & trg_tri_mask
        return trg_mask


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
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
