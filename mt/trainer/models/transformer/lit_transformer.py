import math

import torch
from torch import nn
import torch.nn.functional as F

import pytorch_lightning as pl
import sacrebleu

from mt.helpers import print_translations
from mt.trainer.models.transformer.transformer import Transformer


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
        self.model = Transformer(self.src_tok.get_vocab_size(), self.trg_tok.get_vocab_size(),
                                 src_tok=self.src_tok, trg_tok=self.trg_tok)

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

        # Get output
        output, _ = self.model(src, src_mask, trg[:, :-1], trg_mask[:, :-1])

        # Reshape output / target
        # Let's assume that after the <eos> everything has be predicted as <pad>,
        # and then, we will ignore the pads in the CrossEntropy
        output = output.contiguous().view(-1, trg_vocab)  # (B, L, vocab) => (B*L, vocab)
        trg = trg[:, 1:].contiguous().view(-1)  # Remove <sos> and reshape to vector (B*L)
        ##############################

        # Compute loss
        loss = self.criterion(output, trg)

        # For debugging
        if self.show_translations:
            # We need to reshape the output to get the maximum probabilities for earch batch and position
            # We subtract 1 to the max_len due to the <sos> removal
            idxs = torch.argmax(output.detach().view(batch_size, trg_max_len - 1, trg_vocab), dim=2)

            src_dec = self.src_tok.decode(src)
            hyp_dec = self.trg_tok.decode(idxs.view(batch_size, -1))
            ref_dec = self.trg_tok.decode(trg.detach().view(batch_size, -1))
            print_translations(hypothesis=hyp_dec, references=ref_dec, source=src_dec, limit=1)

        # Logging to TensorBoard by default
        self.log(f'{prefix}_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'{prefix}_ppl', math.exp(loss), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx, prefix="val"):
        # Run one mini-batch
        src, src_mask, trg, trg_mask = batch
        batch_size = src.shape[0]

        # Get output
        output, _ = self.model(src, src_mask, trg[:, :-1], trg_mask[:, :-1])

        # Reshape output / target
        # Let's assume that after the <eos> everything has be predicted as <pad>,
        # and then, we will ignore the pads in the CrossEntropy
        output = output.contiguous().view(-1, output.shape[-1])  # (B, L, vocab) => (B*L, vocab)
        trg = trg[:, 1:].contiguous().view(-1)  # Remove <sos> and reshape to vector (B*L)
        ##############################

        # Compute loss
        loss = self.criterion(output, trg)

        # For debugging
        if self.show_translations:
            src_dec = self.src_tok.decode(src)
            hyp_dec = self.trg_tok.decode(torch.argmax(output.detach(), dim=1).reshape(batch_size, -1))
            ref_dec = self.trg_tok.decode(trg.detach().reshape(batch_size, -1))
            print_translations(hypothesis=hyp_dec, references=ref_dec, source=src_dec, limit=1)

        # Logging to TensorBoard by default
        self.log(f'{prefix}_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'{prefix}_ppl', math.exp(loss), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx, prefix="test"):
        pass


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
