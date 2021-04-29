import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from mt import helpers


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.pe[:, :x.size(1), :]


class Encoder(nn.Module):

    def __init__(self, input_dim, d_model, n_layers, n_heads, dff, dropout, max_length):
        super().__init__()
        self.max_length = max_length

        self.tok_embedding = nn.Embedding(input_dim, d_model)  # Vocab => emb
        self.pos_embedding = PositionalEncoding(d_model, max_length)  # Pos => emb_pos
        # self.pos_embedding = nn.Embedding(max_length, d_model)  # Pos => emb_pos

        self.layers = nn.ModuleList([EncoderLayer(d_model,
                                                  n_heads,
                                                  dff,
                                                  dropout) for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(d_model)

    def forward(self, src, src_mask):
        batch_size = src.shape[0]
        src_len = src.shape[1]
        assert src_len <= self.max_length

        # # Initial positions: 0,1,2,... for each sample
        pos = torch.arange(0, src_len, device=src.device).unsqueeze(0).repeat(batch_size, 1)  # (B, src_len)

        # Mix token embeddings and positional embeddings
        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))  # (B, src_len, d_model)

        for layer in self.layers:
            src = layer(src, src_mask)  # (B, src_len, d_model)

        return src


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dff, dropout):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.ff_layer_norm = nn.LayerNorm(d_model)

        self.self_attention = MultiHeadAttentionLayer(d_model, n_heads, dropout)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(d_model, dff, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        # Multi-head attention
        _src, _ = self.self_attention(src, src, src, src_mask)  # (B, L_enc, d_model)
        src = self.self_attn_layer_norm(src + self.dropout(_src))  # (B, L_enc, d_model)

        # Feedforward
        _src = self.positionwise_feedforward(src)  # (B, L_enc, d_model)
        src = self.ff_layer_norm(src + self.dropout(_src))  # (B, L_enc, d_model)
        return src


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)

        self.fc_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        # Self-attention (query = key = value = x)
        Q = self.fc_q(query)  # (B, L, d_model) => (B, L, d_model)
        K = self.fc_k(key)
        V = self.fc_v(value)

        # From (B, len, dim) => (B, len, n_heads, head_dim) => (B, n_heads, len, head_dim)
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Get un-normalized attention.
        # Transpose Keys => (q_len, head_dim) x (head_dim, k_len) = (q_len, k_len)
        K_t = K.permute(0, 1, 3, 2)
        energy = torch.matmul(Q, K_t) / self.scale

        # Ignore pads
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e18)

        # Normalize attention
        attention = torch.softmax(energy, dim=-1)

        # Encode input with attention (k_len == v_len)
        x = torch.matmul(self.dropout(attention), V)  # [..., q_len, k_len] x [..., v_len, head dim]

        # Go back to the input size
        x = x.permute(0, 2, 1, 3).contiguous()  # (B, n_heads, len, head_dim) => (B, len, n_heads, head_dim)
        x = x.view(batch_size, -1, self.d_model)  # (B, len, n_heads, head_dim) => (B, len, d_model)

        # Linear
        x = self.fc_o(x)  # (..., d_model) => (..., d_model)
        return x, attention


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, d_model, pf_dim, dropout):
        super().__init__()

        self.fc_1 = nn.Linear(d_model, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(torch.relu(self.fc_1(x)))  # Expand
        x = self.fc_2(x)  # Compress
        return x


class Decoder(nn.Module):
    def __init__(self, output_dim, d_model, n_layers, n_heads, dff, dropout, max_length):
        super().__init__()

        self.max_length = max_length

        self.tok_embedding = nn.Embedding(output_dim, d_model)
        self.pos_embedding = PositionalEncoding(d_model, max_length)  # Pos => emb_pos
        # self.pos_embedding = nn.Embedding(max_length, d_model)  # This limits decoding length at testing

        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, dff, dropout)
                                     for _ in range(n_layers)])

        self.fc_out = nn.Linear(d_model, output_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(d_model)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        assert trg_len <= self.max_length

        # # Initial positions: 0,1,2,... for each sample
        # device = trg.device
        pos = torch.arange(0, trg_len, device=trg.device).unsqueeze(0).repeat(batch_size, 1)#.to(device)

        # Mix token embeddings and positional embeddings
        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))

        attention = None
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)

        output = self.fc_out(trg)  # (B, L, d_model) => (B, L, vocab)
        return output, attention


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dff, dropout):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.enc_attn_layer_norm = nn.LayerNorm(d_model)
        self.ff_layer_norm = nn.LayerNorm(d_model)

        self.self_attention = MultiHeadAttentionLayer(d_model, n_heads, dropout)
        self.encoder_attention = MultiHeadAttentionLayer(d_model, n_heads, dropout)

        self.positionwise_feedforward = PositionwiseFeedforwardLayer(d_model, dff, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # Self-attention (target + mask)
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)  # (B, L_dec, d_model)
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))  # (B, L_dec, d_model)

        # Encoder attention
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)   # (B, L_dec, d_model), # (B, nheads, L_dec, L_enc)
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))  # (B, L_dec, d_model)

        # Position-wise feedforward
        _trg = self.positionwise_feedforward(trg)  # (B, L_dec, d_model)
        trg = self.ff_layer_norm(trg + self.dropout(_trg))   # (B, L_dec, d_model)

        return trg, attention


class Transformer(nn.Module):

    def __init__(self, d_model=256,
                 enc_layers=3, dec_layers=3,
                 enc_heads=8, dec_heads=8,
                 enc_dff_dim=512, dec_dff_dim=512,
                 enc_dropout=0.1, dec_dropout=0.1,
                 max_src_len=2000, max_trg_len=2000, src_tok=None, trg_tok=None):
        super().__init__()
        self.src_tok = src_tok
        self.trg_tok = trg_tok

        self.max_src_len = self.src_tok.max_length if self.src_tok else max_src_len
        self.max_trg_len = self.trg_tok.max_length if self.trg_tok else max_trg_len

        self.encoder = Encoder(self.src_tok.get_vocab_size(), d_model, enc_layers, enc_heads, enc_dff_dim, enc_dropout, self.max_src_len)
        self.decoder = Decoder(self.trg_tok.get_vocab_size(), d_model, dec_layers, dec_heads, dec_dff_dim, dec_dropout, self.max_trg_len)

    def make_src_mask(self, src_mask):
        # Extend dimensions
        src_mask = src_mask.unsqueeze(1).unsqueeze(2)  #  (B, n_heads=1, seq_len=1, seq_len)
        return src_mask

    def make_trg_mask(self, trg_mask):
        # Extend dimensions
        trg_mask = trg_mask.unsqueeze(1).unsqueeze(2)  #  (B, n_heads=1, seq_len=1, seq_len)

        # Diagonal matrix to hide next token (LxL)
        trg_len = trg_mask.shape[3]  # target (max) length
        trg_tri_mask = torch.tril(torch.ones((trg_len, trg_len), device=trg_mask.device)).bool()

        # Add pads to the diagonal matrix (LxL)&Pad
        # This is automatically broadcast (B, 1, 1, L) & (L, L) => (B, 1, L, L)
        trg_mask = trg_mask & trg_tri_mask
        return trg_mask

    def forward(self, src, src_mask, trg, trg_mask):
        # Process masks
        src_mask = self.make_src_mask(src_mask)
        trg_mask = self.make_trg_mask(trg_mask)

        # For debugging
        # source = self.src_tok.decode(src)
        # reference = self.trg_tok.decode(trg)
        # helpers.print_translations(source, reference)

        # Encoder-Decoder
        enc_src = self.encoder(src, src_mask)
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        return output, attention

    def decode_word(self, enc_src, src_mask, trg_indexes):
        # Get predicted words (all)
        trg_tensor = torch.tensor(trg_indexes, dtype=torch.int, device=enc_src.device).unsqueeze(0)  # (1, 1->L)
        trg_mask = self.make_trg_mask(trg_tensor)  # (B, n_heads, L, L)

        with torch.no_grad():
            # Inputs: source + current translation
            output, last_attention = self.decoder(trg_tensor, enc_src, trg_mask, src_mask)  # (B, L, vocab), (B, nheads, L_enc, L_dec)

        # Find top k words from the output vocabulary
        return output, last_attention

    def translate_batch(self, src, src_mask, sos_idx, eos_idx, max_length=150, beam_width=3):
        # Build source mask
        src_mask = self.make_src_mask(src_mask)

        # Run encoder
        with torch.no_grad():
            enc_src = self.encoder(src, src_mask)

        # Set fist word (<sos>)
        batch_size = len(src)
        final_candidates = []
        for i in range(batch_size):  # Samples to translate
            candidates = [([sos_idx], 0.0)]  # (ids, probability (unnormalized))

            while True:
                # Expand each candidate
                tmp_candidates = []
                modified = False
                for idxs, score in candidates:
                    # Check if the EOS has been reached, or the maximum length exceeded
                    if idxs[-1] == eos_idx or len(idxs) >= max_length:
                        continue
                    else:
                        modified = True

                    # Get next word probabilities (the decoder returns one output per target-input)
                    next_logits, _ = self.decode_word(enc_src[i].unsqueeze(0), src_mask[i].unsqueeze(0), idxs)
                    next_logits = next_logits.squeeze(0)[-1]  # Ignore batch (Batch always 1); and get last word

                    # Get top k indexes (by score) to reduce the memory consumption
                    new_scores = score + F.log_softmax(next_logits)  # Previous score + new
                    top_idxs_i = torch.argsort(new_scores, descending=True)[:beam_width]  # tmp

                    # Add new candidates
                    new_candidates = [(idxs + [int(idx)], float(new_scores[int(idx)])) for idx in top_idxs_i]
                    tmp_candidates += new_candidates

                # Check if there has been any change
                if modified:
                    # Normalize probabilities, sort in descending order and select top k
                    candidates = sorted(tmp_candidates, key=lambda x: x[1]/len(x[0]), reverse=True)
                    candidates = candidates[:beam_width]
                else:
                    final_candidates.append(candidates)
                    break
        return final_candidates
