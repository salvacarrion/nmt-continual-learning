import math
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder
import torch.nn.functional as F
from einops import rearrange

from mt import helpers


class TransformerModel(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 src_tok=None, trg_tok=None):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.d_model = d_model

        self.pos_enc = PositionalEncoding(d_model=d_model, dropout=dropout)

        # Enc emb
        self.src_tok = src_tok
        self.src_tok_embedding = nn.Embedding(self.src_tok.get_vocab_size(), d_model)

        # Dec emb
        self.trg_tok = trg_tok
        self.trg_tok_embedding = nn.Embedding(self.trg_tok.get_vocab_size(), d_model)

        # Core
        self.transformer_model = nn.Transformer(d_model=d_model, nhead=nhead,
                                                num_encoder_layers=num_encoder_layers,
                                                num_decoder_layers=num_decoder_layers,
                                                dim_feedforward=dim_feedforward, dropout=dropout)
        self.fc = nn.Linear(d_model, self.trg_tok.get_vocab_size())

        self.init_weights()

    def forward(self, src, trg, src_key_padding_mask, trg_key_padding_mask, memory_key_padding_mask, trg_mask):
        # # For debugging
        # source = self.src_tok.decode(src)
        # reference = self.trg_tok.decode(trg)
        # helpers.print_translations(source, reference)

        # Reverse the shape of the batches from (num_sentences, num_tokens_in_each_sentence)
        src = rearrange(src, 'n s -> s n')
        trg = rearrange(trg, 'n t -> t n')

        # Process src/trg
        src = self.pos_enc(self.src_tok_embedding(src) * math.sqrt(self.d_model))
        trg = self.pos_enc(self.trg_tok_embedding(trg) * math.sqrt(self.d_model))

        # Get output
        output = self.transformer_model(src, trg, tgt_mask=trg_mask, src_key_padding_mask=src_key_padding_mask,
                                        tgt_key_padding_mask=trg_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        # Rearrange to batch-first
        output = rearrange(output, 't n e -> n t e')

        # Run the output through an fc layer to return values for each token in the vocab
        return self.fc(output)

    def init_weights(self):
        # Use Xavier normal initialization in the transformer
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=2000):
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
        # [sequence length, batch size, embed dim]
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
