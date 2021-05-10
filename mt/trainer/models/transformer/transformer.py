import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class EncoderInput(nn.Module):

    def __init__(self, input_dim, d_model, dropout, max_length, static_pos_emb=True):
        super().__init__()
        self.max_length = max_length

        self.tok_embedding = nn.Embedding(input_dim, d_model)  # Vocab => emb
        if static_pos_emb:
            self.pos_embedding = PositionalEncoding(d_model, max_length)  # Pos => emb_pos
        else:
            self.pos_embedding = nn.Embedding(max_length, d_model)  # This limits decoding length at testing

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(d_model)

    def forward(self, src):
        batch_size, src_len = src.shape[0], src.shape[1]
        assert src_len <= self.max_length

        # # Initial positions: 0,1,2,... for each sample
        pos = torch.arange(0, src_len, device=src.device).unsqueeze(0).repeat(batch_size, 1)  # (B, src_len)

        # Mix token embeddings and positional embeddings
        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))  # (B, src_len, d_model)
        return src


class Encoder(nn.Module):

    def __init__(self, d_model, n_layers, n_heads, dff, dropout):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model,
                                                  n_heads,
                                                  dff,
                                                  dropout) for _ in range(n_layers)])

    def forward(self, src, src_mask):
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


class DecoderInput(nn.Module):
    def __init__(self, output_dim, d_model, dropout, max_length, static_pos_emb=True):
        super().__init__()
        self.max_length = max_length

        self.tok_embedding = nn.Embedding(output_dim, d_model)
        if static_pos_emb:
            self.pos_embedding = PositionalEncoding(d_model, max_length)  # Pos => emb_pos
        else:
            self.pos_embedding = nn.Embedding(max_length, d_model)  # This limits decoding length at testing

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(d_model)

    def forward(self, trg):
        batch_size, trg_len = trg.shape[0], trg.shape[1]
        assert trg_len <= self.max_length

        # # Initial positions: 0,1,2,... for each sample
        # device = trg.device
        pos = torch.arange(0, trg_len, device=trg.device).unsqueeze(0).repeat(batch_size, 1)

        # Mix token embeddings and positional embeddings
        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
        return trg


class Decoder(nn.Module):
    def __init__(self, d_model, n_layers, n_heads, dff, dropout):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, dff, dropout)
                                     for _ in range(n_layers)])

    def forward(self, trg, enc_src, trg_mask, src_mask):
        attention = None
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)

        return trg, attention


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
                 max_src_len=2000, max_trg_len=2000,
                 src_tok=None, trg_tok=None,
                 static_pos_emb=True):
        super().__init__()
        # Tokenizers
        self.src_tok = src_tok
        self.trg_tok = trg_tok

        # Define enc/dec input
        self.enc_input = EncoderInput(self.src_tok.get_vocab_size(), d_model, enc_dropout, max_src_len, static_pos_emb)
        self.dec_input = DecoderInput(self.trg_tok.get_vocab_size(), d_model, dec_dropout, max_trg_len, static_pos_emb)

        # Transformer (enc/dec)
        self.encoder = Encoder(d_model, enc_layers, enc_heads, enc_dff_dim, enc_dropout)
        self.decoder = Decoder(d_model, dec_layers, dec_heads, dec_dff_dim, dec_dropout)

        # Decouple vocabulary from transformer
        self.fc_out = nn.Linear(d_model, self.trg_tok.get_vocab_size())

        # Initialize
        #self.apply(initialize_weights)

    def forward(self, src, src_mask, trg, trg_mask):
        # Process masks
        src_mask = self.make_src_mask(src_mask)
        trg_mask = self.make_trg_mask(trg_mask)

        # Encoder
        src = self.enc_input(src)
        enc_src = self.encoder(src, src_mask)

        # Decoder
        trg = self.dec_input(trg)
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        output = self.fc_out(output)  # (B, L, d_model) => (B, L, vocab)

        return output, attention

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

    def translate_batch(self, src, max_length=100, beam_width=1):
        self.eval()

        # Build mask
        src_mask = (src != self.src_tok.word2idx[self.src_tok.PAD_WORD])
        src_mask = self.make_src_mask(src_mask)

        # Encoder
        with torch.no_grad():
            src_tensor = self.enc_input(src)
            enc_src = self.encoder(src_tensor, src_mask)

        # Prepare target inputs (sos)
        TRG_SOS_IDX = self.trg_tok.word2idx[self.trg_tok.SOS_WORD]
        TRG_EOS_IDX = self.trg_tok.word2idx[self.trg_tok.EOS_WORD]
        trg = torch.LongTensor([[TRG_SOS_IDX] for _ in range(len(src_tensor))]).to(enc_src.device)
        beam_probs = torch.ones((len(src_tensor), beam_width)).to(enc_src.device)
        eos_found = torch.zeros((len(src_tensor), beam_width)).bool().to(enc_src.device)

        for i in range(max_length):
            new_idxs = []

            for b in range(beam_width):
                _trg = trg[:, b, :] if trg.ndim > 2 else trg

                # Build mask
                trg_mask = (_trg != self.trg_tok.word2idx[self.trg_tok.PAD_WORD])
                trg_mask = self.make_trg_mask(trg_mask)

                # Encode target tensor
                with torch.no_grad():
                    trg_tensor = self.dec_input(_trg)
                    output, attention = self.decoder(trg_tensor, enc_src, trg_mask, src_mask)
                    output = self.fc_out(output)  # (B, L, d_model) => (B, L, vocab)
                    output = F.softmax(output, dim=2)  # (B, L, d_model) => (B, L, vocab)

                # Get top-beam candidates per batch (minor optimization)
                probs, idxs = output[:, -1].sort(dim=1, descending=True)
                probs = probs[:, :beam_width]
                idxs = idxs[:, :beam_width]

                # P(old_beams) * P(new_token)
                prob = beam_probs[:, b].unsqueeze(1) * probs  # / len  Normalize
                _trg = trg[:, b, :].unsqueeze(1) if trg.ndim > 2 else trg.unsqueeze(1)
                idxs = torch.cat([_trg.repeat(1, beam_width, 1), idxs.unsqueeze(2)], dim=2)
                new_idxs.append([idxs, prob])

                if trg.ndim == 2:  # The first input has beam=1 (<sos>)
                    break

            # Get top-k: [a, b, c], [d, e, f], [g, h, i] => [b,f, i]
            idxs = torch.cat([x[0] for x in new_idxs], dim=1)
            probs = torch.cat([x[1] for x in new_idxs], dim=1)
            lengths = torch.argmax((idxs==TRG_EOS_IDX).float(), dim=2)+1  # Add "1" because indices start at 0
            _, probidxs = torch.topk(probs/lengths, beam_width)  # Get top-k with normalized probs
            beam_probs = torch.cat([x[y].unsqueeze(0) for x, y in zip(probs, probidxs)])  # Add unnormalized probs
            trg = torch.cat([x[y].unsqueeze(0) for x, y in zip(idxs, probidxs)])

            # Check if all EOS has been found
            all_eos = bool(torch.all((lengths != 1).bool()))  # "!=1" because indices start at "0"
            if all_eos:
                break
        # Split sentences by beam search
        trg = [trg[:, i, :] for i in range(beam_width)]
        probs = [beam_probs[:, i] for i in range(beam_width)]
        return trg, probs, attention

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)
