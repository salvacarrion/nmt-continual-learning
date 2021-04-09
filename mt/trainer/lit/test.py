import os
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from mt.preprocess import DATASETS_PATH, utils
from mt import helpers
from mt.trainer.models.transformer.lit_transformer import LitTransformer, init_weights

np.random.seed(123)
pl.seed_everything(123)

logger = TensorBoardLogger('../../logs', name='transformer')
checkpoint_path = "/home/salvacarrion/Documents/Programming/Python/nmt-continual-learning/mt/logs/transformer/version_1/checkpoints/tmp|health-epoch=09-train_loss=5.64.ckpt"


# Use zero workers when debugging to avoid freezing
def evaluate_model(datapath, src, trg, domain, batch_size=32//4, max_tokens=4000//4, num_workers=0):
    # Load tokenizers
    lt_src, lt_trg = helpers.get_tokenizers(os.path.join(datapath, "bpe"), src, trg, use_fastbpe=True)  # use_fastbpe != apply_fastbpe

    # Load dataset
    datasets = helpers.load_dataset(os.path.join(datapath, "bpe"), src, trg, splits=["test"])

    # Prepare data loaders
    test_loader = helpers.build_dataloader(datasets["test"], lt_src, lt_trg, apply_bpe=False,
                                           batch_size=batch_size, max_tokens=max_tokens, num_workers=num_workers,
                                           shuffle=False)

    # Instantiate model from checkpoint
    # model = LitTransformer.load_from_checkpoint(checkpoint_path, lt_src=lt_src, lt_trg=lt_trg)
    model = LitTransformer(lt_src, lt_trg)

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            monitor='val_loss',
            filename='transformer-{epoch:02d}-{val_loss:.2f}',
            save_top_k=3,
            mode='min',
        ),
        EarlyStopping(monitor='val_loss')
    ]

    # Train
    trainer = pl.Trainer(min_epochs=1, max_epochs=10, gpus=1,
                         check_val_every_n_epoch=1,

                         # Shorten epochs
                         limit_train_batches=1.0,
                         limit_val_batches=1.0,

                         gradient_clip_val=1.0,
                         # stochastic_weight_avg=True,
                         callbacks=callbacks, logger=logger)

    # Perform training
    trainer.test(model, test_loader)

    print("Done!")


if __name__ == "__main__":
    # Get all folders in the root path
    # datasets = [os.path.join(DATASETS_PATH, name) for name in os.listdir(DATASETS_PATH) if os.path.isdir(os.path.join(DATASETS_PATH, name))]
    datasets = [os.path.join(DATASETS_PATH, "tmp|health_es-en")]
    for dataset in datasets:
        domain, (src, trg) = utils.get_dataset_ids(dataset)
        fname_base = f"{domain}_{src}-{trg}"
        print(f"Training model ({fname_base})...")

        # Train model
        evaluate_model(dataset, src, trg, domain)

