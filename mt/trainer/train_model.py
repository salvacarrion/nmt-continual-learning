import os
import numpy as np

import pytorch_lightning as pl

from mt.models.lit_transformer import LitTransformer
from mt.trainer import helpers

np.random.seed(123)


def train_model(datapath, src, trg, domain, batch_size=32):
    # Load tokenizers
    lt_src, lt_trg = helpers.get_tokenizers(os.path.join(datapath, "tok"), src, trg)

    # Load dataset
    datasets = helpers.load_dataset(os.path.join(datapath, "clean"), src, trg, splits=["val", "test"])

    # Prepare data loaders
    train_loader = helpers.build_dataloader(datasets["test"], lt_src, lt_trg, batch_size=batch_size, num_workers=0)
    val_loader = helpers.build_dataloader(datasets["val"], lt_src, lt_trg, batch_size=batch_size, num_workers=0)

    # Instantiate model
    pad_idx = lt_trg.tokenizer.token_to_id(lt_trg.PAD_WORD)
    model = LitTransformer(lt_src.tokenizer.get_vocab_size(), lt_trg.tokenizer.get_vocab_size(), pad_idx)

    # Train
    trainer = pl.Trainer(max_epochs=1, gpus=1)
    trainer.fit(model, train_loader, val_loader)

    print("Done!")
