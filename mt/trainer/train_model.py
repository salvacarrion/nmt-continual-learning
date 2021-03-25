import os
import numpy as np
from mt.trainer import helpers

np.random.seed(123)


def train_model(datapath, src, trg, domain):
    # Load dataset
    datasets = helpers.load_dataset(os.path.join(datapath, "clean"), src, trg, splits=["val", "test"])

    # Load tokenizers
    lt_src, lt_trg = helpers.get_tokenizers(os.path.join(datapath, "tok"), src, trg)

    # Instantiate model
    asdas = 3
