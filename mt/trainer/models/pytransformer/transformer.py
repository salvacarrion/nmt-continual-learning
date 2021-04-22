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

    def forward(self, src, trg, src_key_padding_mask, trg_key_padding_mask, memory_key_padding_mask):
        # # For debugging
        # source = self.src_tok.decode(src)
        # reference = self.trg_tok.decode(trg)
        # helpers.print_translations(source, reference)

        trg_mask = self.gen_nopeek_mask(trg.shape[1]).to(src.device)  # To not look tokens ahead

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

    def gen_nopeek_mask(self, length):
        """
         Returns the nopeek mask
                 Parameters:
                         length (int): Number of tokens in each sentence in the target batch
                 Returns:
                         mask (arr): tgt_mask, looks like [[0., -inf, -inf],
                                                          [0., 0., -inf],
                                                          [0., 0., 0.]]
         """
        mask = rearrange(torch.triu(torch.ones(length, length)) == 1, 'h w -> w h')
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

        return mask

    def decode_word(self, memory, memory_key_padding_mask, trg_indexes):
        # Get predicted words (all)
        trg = torch.tensor(trg_indexes, dtype=torch.int, device=memory.device).unsqueeze(0)  # (1, 1->L)
        trg_mask = self.gen_nopeek_mask(trg.shape[1]).to(memory.device)  # To not look tokens ahead
        tgt_key_padding_mask = torch.zeros(trg.shape, device=trg.device).bool()

        with torch.no_grad():
            # Reverse the shape of the batches from (num_sentences, num_tokens_in_each_sentence)
            trg = rearrange(trg, 'n t -> t n')

            # Process src/trg
            trg = self.pos_enc(self.trg_tok_embedding(trg) * math.sqrt(self.d_model))

            # Get next word probabilities (the decoder returns one output per target-input)
            output = self.transformer_model.decoder(tgt=trg, memory=memory,
                                                    tgt_mask=trg_mask, memory_mask=None,
                                                    tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
            # Rearrange to batch-first
            output = rearrange(output, 't n e -> n t e')

            # Run the output through an fc layer to return values for each token in the vocab
            output = self.fc(output)
        return output

    def translate_batch(self, src, src_key_padding_mask, max_length=100, beam_width=1):
        sos_idx = self.trg_tok.word2idx[self.trg_tok.SOS_WORD]
        eos_idx = self.trg_tok.word2idx[self.trg_tok.EOS_WORD]

        # Run encoder
        with torch.no_grad():
            # Reverse the shape of the batches from (num_sentences, num_tokens_in_each_sentence)
            src = rearrange(src, 'n s -> s n')
            # Process src/trg
            src = self.pos_enc(self.src_tok_embedding(src) * math.sqrt(self.d_model))
            memory = self.transformer_model.encoder(src, mask=None, src_key_padding_mask=src_key_padding_mask)

        # Set fist word (<sos>)
        batch_size = src.size(1)
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
                    next_logits = self.decode_word(memory=memory[:, i, :].unsqueeze(1),
                                                     memory_key_padding_mask=src_key_padding_mask[i].unsqueeze(0),
                                                     trg_indexes=idxs)
                    next_logits = next_logits.squeeze(0)[-1]  # Ignore batch (Batch always 1); and get last word

                    # Get top k indexes (by score) to reduce the memory consumption
                    new_scores = score + F.log_softmax(next_logits, dim=-1)  # Previous score + new
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
