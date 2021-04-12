import math

import torch
from torch import nn
import torch.nn.functional as F

import pytorch_lightning as pl
import sacrebleu

from mt.helpers import print_translations
from mt.trainer.models.transformer_old.transformer import Seq2Seq, Encoder, Decoder


def init_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


class LitTransformer(pl.LightningModule):

    def __init__(self, lt_src, lt_trg):
        super().__init__()

        # Save tokenizers
        self.src_tok = lt_src
        self.trg_tok = lt_trg
        self.show_translations = False
        self.batch_size = 32
        self.learning_rate = 1e-4

        # Model params
        hid_dim = 256
        enc_layers = 3
        dec_layers = 3
        enc_heads = 8
        dec_heads = 8
        enc_pf_dim = 512
        dec_pf_dim = 512
        enc_dropout = 0.1
        dec_dropout = 0.1
        self.max_src_len = self.src_tok.max_length if self.src_tok else 200
        self.max_trg_len = self.trg_tok.max_length if self.trg_tok else 200

        src_vocab, trg_vocab = self.src_tok.get_vocab_size(), self.trg_tok.get_vocab_size()
        self.enc = Encoder(src_vocab, hid_dim, enc_layers, enc_heads, enc_pf_dim, enc_dropout, self.max_src_len)
        self.dec = Decoder(trg_vocab, hid_dim, dec_layers, dec_heads, dec_pf_dim, dec_dropout, self.max_trg_len)
        self.model = Seq2Seq(self.enc, self.dec, self.src_tok, self.trg_tok)

        # Initialize weights
        self.model.apply(init_weights)

        # Set loss (ignore when the target token is <pad>)
        self.pad_idx = self.trg_tok.word2idx[lt_trg.PAD_WORD]
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_idx)

    def forward(self, x):
        # Inference
        return x

    def training_step(self, batch, batch_idx, prefix="train"):
        # Run one mini-batch
        src, src_mask, trg, trg_mask = batch
        batch_size = src.shape[0]
        src_vocab, trg_vocab = self.src_tok.get_vocab_size(), self.trg_tok.get_vocab_size()
        src_max_len, trg_max_len = src.shape[1], trg.shape[1]

        ##############################
        # Feed input
        output, _ = self.model(src, trg[:, :-1])  # Ignore <eos> as input for trg

        # Reshape output / target
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)  # (B, L, vocab) => (B*L, vocab)
        trg = trg[:, 1:].contiguous().view(-1)  # Remove <sos> and reshape to vector (B*L)
        ##############################

        # Compute loss
        loss = self.criterion(output, trg)

        # For debugging
        if self.show_translations:
            with torch.no_grad():
                src_dec = self.src_tok.decode(src)
                hyp_dec = self.trg_tok.decode(torch.argmax(output.detach().reshape(batch_size, trg_max_len-1, trg_vocab), dim=2).reshape(batch_size, -1))
                ref_dec = self.trg_tok.decode(trg.detach().reshape(batch_size, -1))
                print_translations(hypothesis=hyp_dec, references=ref_dec, source=src_dec, limit=1)

        # Logging to TensorBoard by default
        self.log(f'{prefix}_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'{prefix}_ppl', math.exp(loss), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    # def validation_step(self, batch, batch_idx, prefix="val"):
    #     # Run one mini-batch
    #     src, src_mask, trg, trg_mask = batch
    #     batch_size = src.shape[0]
    #
    #     # Get output
    #     output, _ = self.model(src, src_mask, trg[:, :-1], trg_mask[:, :-1])
    #
    #     # Reshape output / target
    #     # Let's assume that after the <eos> everything has be predicted as <pad>,
    #     # and then, we will ignore the pads in the CrossEntropy
    #     output = output.contiguous().view(-1, output.shape[-1])  # (B, L, vocab) => (B*L, vocab)
    #     trg = trg[:, 1:].contiguous().view(-1)  # Remove <sos> and reshape to vector (B*L)
    #     ##############################
    #
    #     # Compute loss
    #     loss = self.criterion(output, trg)
    #
    #     # For debugging
    #     if self.show_translations:
    #         src_dec = self.src_tok.decode(src)
    #         hyp_dec = self.trg_tok.decode(torch.argmax(output.detach(), dim=1).reshape(batch_size, -1))
    #         ref_dec = self.trg_tok.decode(trg.detach().reshape(batch_size, -1))
    #         print_translations(hypothesis=hyp_dec, references=ref_dec, source=src_dec, limit=1)
    #
    #     # Logging to TensorBoard by default
    #     self.log(f'{prefix}_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    #     self.log(f'{prefix}_ppl', math.exp(loss), on_step=True, on_epoch=True, prog_bar=True, logger=True)
    #     return loss
    #
    # # def test_step(self, batch, batch_idx, prefix="test"):
    # #     self._shared_eval(batch, batch_idx, prefix)
    # #
    # #     # Logging to TensorBoard by default
    # #     self.log(f'{prefix}_blue', 0)
    #
    # def _shared_eval(self, batch, batch_idx, prefix):
    #     src, src_mask, trg, trg_mask = batch
    #
    #     # Get indexes
    #     sos_idx = self.trg_tok.word2idx[self.trg_tok.SOS_WORD]
    #     eos_idx = self.trg_tok.word2idx[self.trg_tok.EOS_WORD]
    #
    #     # Get output
    #     final_candidates = self.model.translate_batch(src, src_mask, sos_idx, eos_idx, self.max_length, self.beam_width)
    #     outputs_ids = [top_trans[0][0] for top_trans in final_candidates]
    #
    #     # Convert ids2words
    #     y_pred = self.trg_tok.decode(outputs_ids)
    #     y_true = self.trg_tok.decode(trg)
    #
    #     # Print translations
    #     if self.show_translations:
    #         print_translations(y_pred, y_true)
    #
    #     # # Compute bleu
    #     # bleu_scores = []
    #     # for sys, ref in zip(y_pred, y_true):
    #     #     bleu = sacrebleu.corpus_bleu([sys], [[ref]])
    #     #     bleu_scores.append(bleu)
    #     # avg_bleu = sum(bleu_scores)/len(bleu_scores)
    #     # return output, losses, metrics

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    # def on_after_backward(self):
    #     global_step = self.global_step
    #     for name, param in self.model.named_parameters():
    #         self.logger.experiment.add_histogram(name, param, global_step)
    #         if param.requires_grad:
    #             self.logger.experiment.add_histogram(f"{name}_grad", param.grad, global_step)