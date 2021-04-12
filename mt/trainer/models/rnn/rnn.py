import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from mt import helpers


class Encoder(nn.Module):

    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

    def forward(self, src, src_len):
        embedded = self.dropout(self.embedding(src))

        total_length = embedded.size(1)  # get the max sequence length (needed for DataParallel)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_len.cpu(), batch_first=True, enforce_sorted=False)

        # packed_outputs is a packed sequence containing all hidden states
        # hidden is now from the final non-padded element in the batch
        packed_outputs, hidden = self.rnn(packed_embedded)

        # outputs is now a non-packed sequence, all hidden states obtained
        #  when the input is a pad token are all zeros
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, total_length=total_length)

        # Concat hiding layers (GRU 1 layer, bidirectional)
        h_fw = hidden[-2, :, :]
        h_bw = hidden[-1, :, :]
        hidden = torch.cat([h_fw, h_bw], dim=1)

        # Transform the double context vector into one.
        # This is done because the decoder is not bidirectional
        hidden = torch.tanh(self.fc(hidden))
        return outputs, hidden


class LuongAttention(nn.Module):

    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)   # Luong
        self.v = nn.Linear(self.dec_hid_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs, mask):
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        # Calculate alignment scores
        decoder_hidden_rep = decoder_hidden.unsqueeze(1)  # (B, D) => (B, 1, D)
        decoder_hidden_rep = decoder_hidden_rep.repeat(1, src_len, 1)  # (B, 1, D) => (B, L, D)
        decoder_hidden_rep = decoder_hidden_rep.permute(1, 0, 2)  # (B, L, D) => (L, B, D)

        # Multi-dimension linears are apply to the last dimension
        pre_scores = self.attn(torch.cat((decoder_hidden_rep, encoder_outputs), dim=2))  # (L, B, E+D) => (L, B, D)

        # Compute scores
        alignment_scores = torch.tanh(pre_scores)
        alignment_scores = self.v(alignment_scores)  # (L, B, D) [x (1, D, 1)] => (L, B, 1)
        alignment_scores = alignment_scores.squeeze(2)  # (L, B, 1) => (L, B)
        alignment_scores = alignment_scores.T  # (L, B) => (B, L)

        # Put -inf where there is padding (The softmax will make them zeros)
        alignment_scores = alignment_scores.masked_fill(mask == 0, -np.inf)

        # Softmax alignment scores
        attn_weights = F.softmax(alignment_scores, dim=1)
        return attn_weights


class Decoder(nn.Module):
    """
    The decoder is going to receive one word at a time. Hence, it is gonna output
    one word after each forward
    """
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        self.out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)  # Luong

    def forward(self, trg, decoder_hidden, encoder_outputs, mask):
        # Target => One word at a time
        embedded = self.dropout(self.embedding(trg.unsqueeze(0)))

        # Compute attention
        attn_weights = self.attention(decoder_hidden, encoder_outputs, mask)

        # Reshape
        attn_weights = attn_weights.unsqueeze(1)  # (B, L) => (B, 1, L)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)  # (L, B, E) => (B, L, E)

        # Context vector: Encoder * Softmax(alignment scores)
        context_vector = torch.bmm(attn_weights, encoder_outputs)  # (B, 1, L) x (B, L, E) => (B, 1, E)
        context_vector = context_vector.permute(1, 0, 2)  # (B, 1, E) => (1, B, E)  // 1 == "L"

        # Concatenate embedding word and context vector
        cat_input = torch.cat([embedded, context_vector], dim=2)  # (1, B, E1), (1, B, E2) => (1, B, E1+E2)
        rnn_output, decoder_hidden = self.rnn(cat_input, decoder_hidden.unsqueeze(0))

        # seq len, n layers and n directions will always be 1 in this decoder, therefore:
        # output = [1, batch size, dec hid dim]
        # hidden = [1, batch size, dec hid dim]
        # this also means that output == hidden
        assert (rnn_output == decoder_hidden).all()

        rnn_output = rnn_output.squeeze(0)
        context_vector = context_vector.squeeze(0)
        embedded = embedded.squeeze(0)

        preout = torch.cat((rnn_output, context_vector, embedded), dim=1)
        prediction = self.out(preout)   # Luong

        return prediction, decoder_hidden.squeeze(0), attn_weights.squeeze(1)


class Seq2Seq(nn.Module):

    def __init__(self, src_vocab_size, trg_vocab_size,
                 enc_emb_dim=256, dec_emb_dim=256,
                 enc_hid_dim=512, dec_hid_dim=512,
                 enc_dropout=0.5, dec_dropout=0.5,
                 tf_ratio=0.5,
                 src_tok=None, trg_tok=None):
        super().__init__()
        self.src_tok = src_tok
        self.trg_tok = trg_tok
        self.tf_ratio = tf_ratio

        self.attn = LuongAttention(enc_hid_dim, dec_hid_dim)
        self.encoder = Encoder(src_vocab_size, enc_emb_dim, enc_hid_dim, dec_hid_dim, enc_dropout)
        self.decoder = Decoder(trg_vocab_size, dec_emb_dim, enc_hid_dim, dec_hid_dim, dec_dropout, self.attn)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, src, src_mask, trg, trg_mask):
        # Run encoder
        src_len = src_mask.sum(dim=1)
        encoder_outputs, hidden = self.encoder(src, src_len)

        # first input to the decoder is the <sos> token
        dec_input = trg[:, 0]  # t=0; Get first word index
        # There is no point in setting output[0] to <sos> since it will be ignored later

        # Store outputs (L, B, trg_vocab)  => Indices
        outputs = torch.zeros(trg.shape[1], trg.shape[0], self.decoder.output_dim, device=trg_mask.device)

        # Iterate over target (max) length
        for t in range(1, trg.shape[1]):
            output, hidden, _ = self.decoder(dec_input, hidden, encoder_outputs, src_mask)
            outputs[t] = output

            # Teacher forcing
            teacher_force = random.random() < self.tf_ratio
            if teacher_force:  # Use actual token
                dec_input = trg[:, t]  # t+1
            else:
                top1 = output.max(1)[1]  # [0]=>values; [1]=>indices
                dec_input = top1  # t+1

        return outputs

    def decode_word(self, enc_outputs, dec_hidden, src_mask, trg_indexes):
        # Get predicted words (all)
        dec_input = torch.tensor(trg_indexes, dtype=torch.int, device=enc_outputs.device)

        with torch.no_grad():
            output, hidden, _ = self.decoder(dec_input, dec_hidden, enc_outputs, src_mask)

        # Find top k words from the output vocabulary
        probs = self.softmax(output)  # (B, L, vocab)
        return probs, hidden

    def translate_batch(self, src, src_mask, sos_idx, eos_idx, max_length=150, beam_width=1):
        # Run encoder
        src_len = src_mask.sum(dim=1)
        enc_outputs, hidden = self.encoder(src, src_len)

        # Set fist word (<sos>)
        batch_size = len(src)
        final_candidates = []
        for i in range(batch_size):  # Samples to translate
            candidates = [([sos_idx], 0.0)]  # (ids, probability (unnormalized))
            dec_hidden = hidden[i].unsqueeze(0)

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
                    next_probs, dec_hidden = self.decode_word(enc_outputs[:, i, :].unsqueeze(1), dec_hidden, src_mask[i].unsqueeze(0), [idxs[-1]])
                    next_probs = next_probs.squeeze(0)  # Ignore batch (Batch always 1); and get last word

                    # Get top k indexes (by score) to reduce the memory consumption
                    new_scores = score + torch.log(next_probs)  # Previous score + new
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
