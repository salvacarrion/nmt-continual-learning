import os
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from mt.models.lit_transformer import LitTransformer
from mt.trainer import helpers

np.random.seed(123)
logger = TensorBoardLogger('tb_logs', name='nmt-cl')


def train_model(datapath, src, trg, domain, batch_size=32, max_tokens=4000, num_workers=0):
    # Load tokenizers
    lt_src, lt_trg = helpers.get_tokenizers(os.path.join(datapath, "bpe"), src, trg, use_fastbpe=True)  # use_fastbpe != apply_fastbpe

    # Load dataset
    datasets = helpers.load_dataset(os.path.join(datapath, "bpe"), src, trg, splits=["train", "val"])

    # Prepare data loaders
    train_loader = helpers.build_dataloader(datasets["train"], lt_src, lt_trg, batch_size=batch_size, max_tokens=max_tokens, num_workers=num_workers)
    val_loader = helpers.build_dataloader(datasets["val"], lt_src, lt_trg, batch_size=batch_size, max_tokens=max_tokens, num_workers=num_workers, shuffle=False)

    # Instantiate model
    model = LitTransformer(lt_src, lt_trg)

    # Callbacks
    callbacks = [
        # ModelCheckpoint(
        #     monitor='val_loss',
        #     filename='transformer-{epoch:02d}-{val_loss:.2f}',
        #     save_top_k=3,
        #     mode='min',
        # ),
        # EarlyStopping(monitor='val_loss')
    ]

    # Train
    trainer = pl.Trainer(min_epochs=1, max_epochs=10, gpus=1,
                         check_val_every_n_epoch=1,

                         # Shorten epochs
                         limit_train_batches=0.1,
                         limit_val_batches=1.0,

                         gradient_clip_val=1.0,
                         # stochastic_weight_avg=True,
                         callbacks=callbacks, logger=logger)
    trainer.fit(model, train_loader, val_loader)

    print("Done!")
