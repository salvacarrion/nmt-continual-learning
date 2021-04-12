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
        self.learning_rate = 1e-3

        # Model params
        self.model = Transformer(self.src_tok.get_vocab_size(), self.trg_tok.get_vocab_size(), src_tok=self.src_tok, trg_tok=self.trg_tok)

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
        output, _ = self.model(src, src_mask, trg[:, :-1], trg_mask[:, :-1])

        # Reshape output / target
        # Let's presume that after the <eos> everything has be predicted as <pad>,
        # and then, we will ignore the pads in the CrossEntropy
        _output = output.contiguous().view(-1, output.shape[-1])  # (B, L, vocab) => (B*L, vocab)
        _trg = trg[:, 1:].contiguous().view(-1)  # Remove <sos> and reshape to vector (B*L)
        ##############################

        # Compute loss
        loss = self.criterion(_output, _trg.type(torch.long))

        # For debugging
        if self.show_translations:
            _src = src.detach()
            _output = torch.argmax(_output.detach(), dim=1).reshape(batch_size, -1)
            _trg = _trg.detach().reshape(batch_size, -1)
            src_dec = self.src_tok.decode(_src)
            hyp_dec = self.trg_tok.decode(_output)
            ref_dec = self.trg_tok.decode(_trg)
            print_translations(hypothesis=hyp_dec, references=ref_dec, source=src_dec, limit=1)

        # Logging to TensorBoard by default
        self.log('train_loss', loss)
        self.log('train_ppl', math.exp(loss))
        return loss

    def validation_step(self, batch, batch_idx):
        # # Run one mini-batch
        # src, src_mask, trg, trg_mask = batch
        # output, losses, metrics, (_output, _trg) = self._batch_step(src, src_mask, trg, trg_mask)
        #
        # # For debugging
        # hyp_dec = self.trg_tok.decode(torch.argmax(_output, dim=1).unsqueeze(0))
        # ref_dec = self.trg_tok.decode(_trg.unsqueeze(0))
        # print_translations(hyp_dec, ref_dec)
        #
        # # Logging to TensorBoard by default
        # self.log('val_loss', losses['loss'])
        # self.log('val_ppl', metrics['ppl'])
        return 0 #losses['loss']

    def test_step(self, batch, batch_idx, max_length=50, beam_width=1):
        src, src_mask, trg, trg_mask = batch

        # Get indexes
        sos_idx = self.trg_tok.word2idx[self.trg_tok.SOS_WORD]
        eos_idx = self.trg_tok.word2idx[self.trg_tok.EOS_WORD]

        # Get output
        final_candidates = self.model.translate_batch(src, src_mask, sos_idx, eos_idx, max_length, beam_width)
        outputs_ids = [top_trans[0][0] for top_trans in final_candidates]

        # Convert ids2words
        y_pred = self.trg_tok.decode(outputs_ids)
        y_true = self.trg_tok.decode(trg)

        # Print translations
        print_translations(y_pred, y_true)

        # # Compute bleu
        # bleu_scores = []
        # for sys, ref in zip(y_pred, y_true):
        #     bleu = sacrebleu.corpus_bleu([sys], [[ref]])
        #     bleu_scores.append(bleu)
        # avg_bleu = sum(bleu_scores)/len(bleu_scores)

        # Logging to TensorBoard by default
        self.log(f'test_blue', 0)
        return 0

    # def _batch_step(self, src, src_mask, trg, trg_mask):
    #     # trg_lengths = trg_mask.sum(dim=1) + 1  # Not needed
    #
    #     # For debugging
    #     # source = self.src_tok.decode(src)
    #     # reference = self.trg_tok.decode(trg)
    #     # print_translations(source, reference)
    #
    #     # Feed input
    #     # src => whole sentence (including <sos>, <eos>)
    #     # src_mask => whole sentence (including <sos>, <eos>)
    #     # trg => whole sentence (except <eos> or use mask to remove it)
    #     # trg_mask => whole sentence except <eos>
    #     # NOTE: I remove the last token from TRG so that the prediction is L-1. This is
    #     # because later we will remove the <sos> from the TRG
    #     output, _ = self.model(src, src_mask, trg[:, :-1], trg_mask[:, :-1])
    #
    #     # Reshape output / target
    #     # Let's presume that after the <eos> everything has be predicted as <pad>,
    #     # and then, we will ignore the pads in the CrossEntropy
    #     output_dim = output.shape[-1]
    #     _output = output.contiguous().view(-1, output_dim)  # (B, L, vocab) => (B*L, vocab)
    #     _trg = trg[:, 1:].contiguous().view(-1)  # Remove <sos> and reshape to vector (B*L)
    #     ##############################
    #
    #     # Compute loss
    #     loss = F.cross_entropy(_output, _trg, ignore_index=self.pad_idx)
    #     # loss = self.criterion(_output, _trg)
    #
    #     # Save loss and metrics
    #     losses = {'loss': loss}
    #     metrics = {'ppl': math.exp(losses['loss'])}
    #     return output, losses, metrics, (_output, _trg)

    # def _shared_eval(self, batch, batch_idx, prefix):
    #     # Run one mini-batch
    #     output, losses, metrics = self._batch_step(batch, batch_idx)
    #     # src, src_mask, trg, trg_mask = batch
    #
    #     # output_trg = torch.argmax(output, 2)
    #     # output_words = self.trg_tok.decode(output_trg)
    #
    #     # Logging to TensorBoard by default
    #     self.log(f'{prefix}_loss', losses['loss'])
    #     self.log(f'{prefix}_ppl', metrics['ppl'])
    #     return output, losses, metrics

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    # def on_after_backward(self):
    #     global_step = self.global_step
    #     for name, param in self.model.named_parameters():
    #         self.logger.experiment.add_histogram(name, param, global_step)
    #         if param.requires_grad:
    #             self.logger.experiment.add_histogram(f"{name}_grad", param.grad, global_step)