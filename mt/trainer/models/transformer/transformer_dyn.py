import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from mt.trainer.models.transformer.transformer import *


class TransformerDyn(nn.Module):

    def __init__(self, d_model=256,
                 enc_layers=3, dec_layers=3,
                 enc_heads=8, dec_heads=8,
                 enc_dff_dim=512, dec_dff_dim=512,
                 enc_dropout=0.1, dec_dropout=0.1,
                 max_src_len=2000, max_trg_len=2000,
                 src_tok=None, trg_tok=None,
                 static_pos_emb=True,
                 encoder_shared=None, decoder_shared=None):
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
        self.encoder_shared = encoder_shared
        self.decoder_shared = decoder_shared

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

    def forward_shared(self, src, src_mask, trg, trg_mask):
        # Process masks
        src_mask = self.make_src_mask(src_mask)
        trg_mask = self.make_trg_mask(trg_mask)

        # Encoder
        src = self.enc_input(src)
        enc_src = self.encoder_shared(src, src_mask)

        # Decoder
        trg = self.dec_input(trg)
        output, attention = self.decoder_shared(trg, enc_src, trg_mask, src_mask)
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
