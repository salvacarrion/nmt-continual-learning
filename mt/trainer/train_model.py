import os
import numpy as np

import pytorch_lightning as pl

from mt.models.transformer import LitTransformer
from mt.trainer import helpers

np.random.seed(123)


def train_model(datapath, src, trg, domain):
    # Load tokenizers
    lt_src, lt_trg = helpers.get_tokenizers(os.path.join(datapath, "tok"), src, trg)

    # Load dataset
    datasets = helpers.load_dataset(os.path.join(datapath, "clean"), src, trg, splits=["val", "test"])

    # Prepare dataloaders
    train_loader = helpers.build_dataloader(datasets["val"], lt_src, lt_trg, batch_size=1, num_workers=0)
    val_loader = helpers.build_dataloader(datasets["test"], lt_src, lt_trg, batch_size=1, num_workers=0)

    # Instantiate model
    model = LitTransformer()

    # Train
    trainer = pl.Trainer(max_epochs=1, gpus=1)
    #trainer.tune(model, train_loader, val_loader)
    trainer.fit(model, train_loader, val_loader)

    print("Done!")
