import math

import torch
from torch import nn
import torch.nn.functional as F

import pytorch_lightning as pl
import sacrebleu

from mt.helpers import print_translations
from mt.trainer.models.rnn.rnn import Seq2Seq


def init_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


class LitRNN(pl.LightningModule):

    def __init__(self, lt_src, lt_trg):
        super().__init__()

        # Save tokenizers
        self.src_tok = lt_src
        self.trg_tok = lt_trg

        # Others
        self.show_translations = False
        self.max_length = 150
        self.beam_width = 3
        self.learning_rate = 1e-3

        # Model params
        self.model = Seq2Seq(self.src_tok.get_vocab_size(), self.trg_tok.get_vocab_size(), src_tok=self.src_tok, trg_tok=self.trg_tok)

        # Initialize weights
        self.model.apply(init_weights)

        # Set loss (ignore when the target token is <pad>)
        self.pad_idx = self.trg_tok.word2idx[lt_trg.PAD_WORD]
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_idx)

    def forward(self, x):
        # Inference
        return x

    def training_step(self, batch, batch_idx):
        # Run one mini-batch
        src, src_mask, trg, trg_mask = batch
        batch_size = src.shape[0]

        # Get output
        output = self.model(src, src_mask, trg, trg_mask)

        # Ignore <sos> token
        _output, _trg = output[1:], trg[:, 1:]

        # Reshape output / target
        _output = _output.contiguous().view(-1, output.shape[-1])  # (L, B, vocab) => (L*B, vocab)
        _trg = _trg.permute(1, 0).reshape(-1)  # (L, B) => (L*B) // We can use class numbers, no need for one-hot encoding
        ##############################

        # Compute loss
        loss = self.criterion(_output, _trg.type(torch.long))

        # For debugging
        if self.show_translations:
            _src = src.detach()
            _output = torch.argmax(_output.detach(), dim=1).reshape(-1, batch_size).permute(1, 0)
            _trg = _trg.detach().reshape(-1, batch_size).permute(1, 0)
            src_dec = self.src_tok.decode(_src)
            hyp_dec = self.trg_tok.decode(_output)
            ref_dec = self.trg_tok.decode(_trg)
            print_translations(hypothesis=hyp_dec, references=ref_dec, source=src_dec, limit=1)

        # Logging to TensorBoard by default
        self.log('train_loss', loss)
        self.log('train_ppl', math.exp(loss))
        return loss

    def validation_step(self, batch, batch_idx, prefix="val"):
        self._shared_eval(batch, batch_idx, prefix)

        # Logging to TensorBoard by default
        self.log(f'{prefix}_blue', 0)

    def test_step(self, batch, batch_idx, prefix="test"):
        self._shared_eval(batch, batch_idx, prefix)

        # Logging to TensorBoard by default
        self.log(f'{prefix}_blue', 0)

    def _shared_eval(self, batch, batch_idx, prefix):
        src, src_mask, trg, trg_mask = batch

        # Get indexes
        sos_idx = self.trg_tok.word2idx[self.trg_tok.SOS_WORD]
        eos_idx = self.trg_tok.word2idx[self.trg_tok.EOS_WORD]

        # Get output
        final_candidates = self.model.translate_batch(src, src_mask, sos_idx, eos_idx, self.max_length, self.beam_width)
        outputs_ids = [top_trans[0][0] for top_trans in final_candidates]

        # Convert ids2words
        y_pred = self.trg_tok.decode(outputs_ids)
        y_true = self.trg_tok.decode(trg)

        # Print translations
        if self.show_translations:
            print_translations(y_pred, y_true)

        # # Compute bleu
        # bleu_scores = []
        # for sys, ref in zip(y_pred, y_true):
        #     bleu = sacrebleu.corpus_bleu([sys], [[ref]])
        #     bleu_scores.append(bleu)
        # avg_bleu = sum(bleu_scores)/len(bleu_scores)
        # return output, losses, metrics

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    # def on_after_backward(self):
    #     global_step = self.global_step
    #     for name, param in self.model.named_parameters():
    #         self.logger.experiment.add_histogram(name, param, global_step)
    #         if param.requires_grad:
    #             self.logger.experiment.add_histogram(f"{name}_grad", param.grad, global_step)