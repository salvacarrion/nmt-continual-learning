import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl


class LitTransformer(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.model = nn.Transformer(nhead=16, num_encoder_layers=12)

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        pass

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        src, src_mask, trg, trg_mask = batch
        assert len(src) == len(src_mask)
        assert len(trg) == len(trg_mask)

        # Predict
        y_hat = self.model(src, trg, src_mask, trg_mask)

        # Logging to TensorBoard by default
        loss = F.nll_loss(y_hat, trg)
        self.log('train_loss', loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


