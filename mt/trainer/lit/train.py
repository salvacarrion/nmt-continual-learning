import os
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from mt.preprocess import DATASETS_PATH, utils
from mt import helpers
from mt.trainer.models.transformer.lit_transformer import LitTransformer, init_weights
from mt.trainer.models.rnn.lit_rnn import LitRNN, init_weights

np.random.seed(123)
pl.seed_everything(123)

MODEL_NAME = "rnn"
logger = TensorBoardLogger('../../logs', name=MODEL_NAME)


# Use zero workers when debugging to avoid freezing
def train_model(datapath, src, trg, domain, batch_size=32//2, max_tokens=4000//2, num_workers=0):
    # Load tokenizers
    bpe_folder = "bpe.8000"
    lt_src, lt_trg = helpers.get_tokenizers(os.path.join(datapath, bpe_folder), src, trg, use_fastbpe=True)  # use_fastbpe != apply_fastbpe

    # Load dataset
    datasets = helpers.load_dataset(os.path.join(datapath, bpe_folder), src, trg, splits=["train", "val", "test"])

    # Prepare data loaders
    train_loader = helpers.build_dataloader(datasets["train"], lt_src, lt_trg, batch_size=batch_size, max_tokens=max_tokens, num_workers=num_workers)
    val_loader = helpers.build_dataloader(datasets["val"], lt_src, lt_trg, batch_size=batch_size, max_tokens=max_tokens, num_workers=num_workers, shuffle=False)
    # test_loader = helpers.build_dataloader(datasets["test"], lt_src, lt_trg, batch_size=batch_size, max_tokens=max_tokens, num_workers=num_workers, shuffle=False)

    # Instantiate model
    print(f"=> Model chosen: '{MODEL_NAME}'")
    if MODEL_NAME == "rnn":
        model = LitRNN(lt_src, lt_trg)
    elif MODEL_NAME == "transformer":
        model = LitTransformer(lt_src, lt_trg)
    else:
        raise ValueError("Unknown model")

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            monitor='train_loss',
            filename=domain+'-{epoch:02d}-{train_loss:.2f}',
            save_top_k=3,
            mode='min',
        ),
        # EarlyStopping(monitor='train_loss')
    ]

    # Train
    trainer = pl.Trainer(
        # min_epochs=1, max_epochs=50,
        gpus=1,
        accumulate_grad_batches=1,
        check_val_every_n_epoch=1,
        gradient_clip_val=1.0,
        overfit_batches=1,  # For debugging
        callbacks=callbacks, logger=logger,
        deterministic=True)

    # Perform training
    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=None)

    # # Perform evaluation
    # trainer.test(model, test_loader)

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
        train_model(dataset, src, trg, domain)
