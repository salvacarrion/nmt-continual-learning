import os
from pathlib import Path

from mt.preprocess import DATASETS_PATH, DATASET_SPLITS_NAME, VOCAB_SIZE
from mt.preprocess import utils

from trainer.tok.lit_tokenizer import LitTokenizer


# Get all folders in the root path
datasets = [os.path.join(DATASETS_PATH, name) for name in os.listdir(DATASETS_PATH) if os.path.isdir(os.path.join(DATASETS_PATH, name))]
# datasets = [os.path.join(DATASETS_PATH, "tmp|health_es-en")]
for dataset in datasets:
    domain, (src, trg) = utils.get_dataset_ids(dataset)
    fname_base = f"{domain}_{src}-{trg}"
    print(f"Processing dataset ({fname_base})...")

    # Create path
    savepath = os.path.join(dataset, "tok")
    Path(savepath).mkdir(parents=True, exist_ok=True)

    # Deine tokenizers
    lt_src = LitTokenizer(padding=False, truncation=False, max_length=5000, lang=src)
    lt_trg = LitTokenizer(padding=False, truncation=False, max_length=5000, lang=trg)

    # Training tokenizer
    print(f"Training vocabs...")
    lt_src.train_vocab([os.path.join(DATASETS_PATH, dataset, DATASET_SPLITS_NAME, f"train.{src}")], vocab_size=VOCAB_SIZE)

    # Sanity check
    # text_src = "Hello world!"
    # text_src_enc = lt_src.tokenizer.encode(text_src)
    # text_src_dec = lt_src.tokenizer.decode(text_src_enc.ids)
    # print(f"Source tokenizer")
    # print(f"\tRaw text: {text_src}")
    # print(f"\tEncoded text: {text_src_enc.tokens}")
    # print(f"\tDecoded text: {text_src_dec}")
    # print("")
    # #
    # lt_trg.train_vocab([os.path.join(DATASETS_PATH, dataset, DATASET_SPLITS_NAME, f"train.{trg}")], vocab_size=VOCAB_SIZE)
    # text_trg = "Hola mundo!"
    # text_trg_enc = lt_trg.tokenizer.encode(text_trg)
    # text_trg_dec = lt_trg.tokenizer.decode(text_trg_enc.ids)
    # print(f"Target tokenizer")
    # print(f"\tRaw text: {text_trg}")
    # print(f"\tEncoded text: {text_trg_enc.tokens}")
    # print(f"\tDecoded text: {text_trg_dec}")
    # print("")

    # Save tokenizers
    lt_src.save_vocab(savepath, prefix=f"tok.{src}")
    lt_trg.save_vocab(savepath, prefix=f"tok.{trg}")

    # # Load tokenizer
    # lt_src.load_vocab(os.path.join(BASE_PATH, "tok", f"tok.{SRC}-vocab.json"),
    #                   os.path.join(BASE_PATH, "tok", f"tok.{SRC}-merges.txt"))
    # lt_trg.load_vocab(os.path.join(BASE_PATH, "tok", f"tok.{TRG}-vocab.json"),
    #                   os.path.join(BASE_PATH, "tok", f"tok.{TRG}-merges.txt"))
asdas = 3
