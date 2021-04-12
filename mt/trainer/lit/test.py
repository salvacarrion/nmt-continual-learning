import os
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from mt.preprocess import DATASETS_PATH, LOGS_PATH, utils
from mt import helpers
from mt.trainer.models.transformer.lit_transformer import LitTransformer, init_weights
from mt.trainer.models.rnn.lit_rnn import LitRNN, init_weights


MODEL_NAME = "transformer"
BPE_FOLDER = "bpe.8000"

np.random.seed(123)
pl.seed_everything(123)


def get_model(model_name, lt_src, lt_trg):
    print(f"=> Model chosen: '{model_name}'")
    if model_name == "rnn":
        model = LitRNN(lt_src, lt_trg)
    elif model_name == "transformer":
        model = LitTransformer(lt_src, lt_trg)
    else:
        raise ValueError("Unknown model")
    return model


# Use zero workers when debugging to avoid freezing
def evaluate_model(datapath, src, trg, model_name, bpe_folder, domain=None, batch_size=32, max_tokens=4096, num_workers=0):
    # Load tokenizers
    lt_src, lt_trg = helpers.get_tokenizers(os.path.join(datapath, bpe_folder), src, trg, use_fastbpe=True)  # use_fastbpe != apply_fastbpe

    # Load dataset
    datasets = helpers.load_dataset(os.path.join(datapath, bpe_folder), src, trg, splits=["train", "val", "test"])

    # Prepare data loaders
    test_loader = helpers.build_dataloader(datasets["test"], lt_src, lt_trg, batch_size=batch_size, max_tokens=max_tokens, num_workers=num_workers, shuffle=False)

    # Instantiate model
    model = get_model(model_name, lt_src, lt_trg)
    model.show_translations = True

    # Train
    trainer = pl.Trainer(min_epochs=1, max_epochs=10, gpus=1,
                         check_val_every_n_epoch=1,

                         # Shorten epochs
                         limit_train_batches=1.0,
                         limit_val_batches=1.0,

                         gradient_clip_val=1.0,
                         # stochastic_weight_avg=True,
                         )

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
        print(f"Testing model ({fname_base})...")

        # Evaluate model
        evaluate_model(dataset, src, trg, model_name=MODEL_NAME, bpe_folder=BPE_FOLDER, domain=domain)

