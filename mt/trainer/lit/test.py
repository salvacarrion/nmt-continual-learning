import os
import numpy as np
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from mt.preprocess import utils
from mt import helpers
from mt import DATASETS_PATH, DATASET_EVAL_NAME
from mt.trainer.models.transformer.lit_transformer import LitTransformer, init_weights
from mt.trainer.models.rnn.lit_rnn import LitRNN, init_weights


MODEL_NAME = "transformer"
tok_folder = "bpe.8000"
CHECKPOINT = "/home/salvacarrion/linux-hdd/Datasets/Scielo/constrained/datasets/tmp|health_es-en/checkpoints/version_0/checkpoints/tmp|health-epoch=10-train_loss=1.48.ckpt"
MAX_LENGTH=100
BEAM_WIDTH=1
np.random.seed(123)
pl.seed_everything(123)


# Use zero workers when debugging to avoid freezing
def evaluate_model(datapath, src, trg, model_name, tok_folder, domain="None", batch_size=32, max_tokens=4096, num_workers=0):
    # Load tokenizers
    src_tok, trg_tok = helpers.get_tokenizers(os.path.join(datapath, tok_folder), src, trg, use_fastbpe=True)  # use_fastbpe != apply_fastbpe

    # Load dataset
    datasets = helpers.load_dataset(os.path.join(datapath, tok_folder), src, trg, splits=["train", "val", "test"])

    # Prepare data loaders
    test_loader = helpers.build_dataloader(datasets["test"], src_tok, trg_tok, batch_size=batch_size, max_tokens=max_tokens, num_workers=num_workers, shuffle=False)

    # Instanciate model
    litmodel = LitTransformer.load_from_checkpoint(src_tok=src_tok, trg_tok=trg_tok, checkpoint_path=CHECKPOINT)

    # Translate
    print("Generating translations...")
    y_pred, y_true = helpers.generate_translations(litmodel.model, trg_tok, test_loader, max_length=MAX_LENGTH, beam_width=BEAM_WIDTH)

    # Create path
    eval_name = domain
    eval_path = os.path.join(datapath, DATASET_EVAL_NAME, eval_name)
    Path(eval_path).mkdir(parents=True, exist_ok=True)

    # Save translations to file
    with open(os.path.join(eval_path, 'hyp.txt'), 'w') as f:
        f.writelines("%s\n" % s for s in y_pred)
    with open(os.path.join(eval_path, 'ref.txt'), 'w') as f:
        f.writelines("%s\n" % s for s in y_true)
    print("Translations written!")

    print("To get BLEU use: 'cat hyp.txt | sacrebleu ref.txt'")
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
        evaluate_model(dataset, src, trg, model_name=MODEL_NAME, tok_folder=tok_folder, domain=domain)

