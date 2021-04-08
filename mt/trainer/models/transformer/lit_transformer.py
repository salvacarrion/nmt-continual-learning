import math

import torch
from torch import nn
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

        # Model params
        self.transformer = Transformer(self.src_tok.get_vocab_size(), self.trg_tok.get_vocab_size())

        # Initialize weights
        self.apply(init_weights)

        # Set loss (ignore when the target token is <pad>)
        pad_idx = self.trg_tok.word2idx[lt_trg.PAD_WORD]
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    def forward(self, x):
        # Inference
        return x

    def training_step(self, batch, batch_idx):
        # Run one mini-batch
        src, src_mask, trg, trg_mask = batch
        output, losses, metrics, _ = self._batch_step(src, src_mask, trg, trg_mask)

        # Logging to TensorBoard by default
        self.log('train_loss', losses['loss'])
        self.log('train_ppl', metrics['ppl'])
        return losses['loss']

    def validation_step(self, batch, batch_idx):
        # Run one mini-batch
        src, src_mask, trg, trg_mask = batch
        output, losses, metrics, (_output, _trg) = self._batch_step(src, src_mask, trg, trg_mask)

        # For debugging
        hyp_dec = self.trg_tok.decode(torch.argmax(_output, dim=1).unsqueeze(0))
        ref_dec = self.trg_tok.decode(_trg.unsqueeze(0))
        print_translations(hyp_dec, ref_dec)

        # Logging to TensorBoard by default
        self.log('val_loss', losses['loss'])
        self.log('val_ppl', metrics['ppl'])
        return losses['loss']

    def test_step(self, batch, batch_idx, max_length=50, beam_width=1):
        src, src_mask, trg, trg_mask = batch

        # Get indexes
        sos_idx = self.trg_tok.word2idx[self.trg_tok.SOS_WORD]
        eos_idx = self.trg_tok.word2idx[self.trg_tok.EOS_WORD]

        # Get output
        final_candidates = self.transformer.translate_batch(src, src_mask, sos_idx, eos_idx, max_length, beam_width)
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

    def _batch_step(self, src, src_mask, trg, trg_mask):
        # trg_lengths = trg_mask.sum(dim=1) + 1  # Not needed

        # For debugging
        # src_dec = self.src_tok.decode(src)
        # trg_dec = self.trg_tok.decode(trg)
        # print_translations(src_dec, trg_dec)

        # Feed input
        # src => whole sentence (including <sos>, <eos>)
        # src_mask => whole sentence (including <sos>, <eos>)
        # trg => whole sentence (except <eos> or use mask to remove it)
        # trg_mask => whole sentence except <eos>
        # NOTE: I remove the last token from TRG so that the prediction is L-1. This is
        # because later we will remove the <sos> from the TRG
        output, _ = self.transformer(src, src_mask, trg[:, :-1], trg_mask[:, :-1])

        # Reshape output / target
        # Let's presume that after the <eos> everything has be predicted as <pad>,
        # and then, we will ignore the pads in the CrossEntropy
        output_dim = output.shape[-1]
        _output = output.contiguous().view(-1, output_dim)  # (B, L, vocab) => (B*L, vocab)
        _trg = trg[:, 1:].contiguous().view(-1)  # Remove <sos> and reshape to vector (B*L)
        ##############################

        # Compute loss and metrics
        losses = {'loss': self.criterion(_output, _trg)}
        metrics = {'ppl': math.exp(losses['loss'])}
        return output, losses, metrics, (_output, _trg)

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

    def configure_optimizers(self, lr=10e-3):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        return optimizer
