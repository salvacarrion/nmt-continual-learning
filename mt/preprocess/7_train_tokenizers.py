import os
from pathlib import Path

from mt import DATASETS_PATH, DATASET_CLEAN_NAME, DATASET_TOK_NAME, VOCAB_SIZE
from mt.preprocess import utils

from trainer.tok.lit_tokenizer import LitTokenizer
from trainer.tok.word_tokenizer import WordTokenizer


tok_model = "wt"

# Get all folders in the root path
datasets = [os.path.join(DATASETS_PATH, name) for name in os.listdir(DATASETS_PATH) if os.path.isdir(os.path.join(DATASETS_PATH, name))]
# datasets = [os.path.join(DATASETS_PATH, "multi30k_de-en")]
for dataset in datasets:
    domain, (src, trg) = utils.get_dataset_ids(dataset)
    fname_base = f"{domain}_{src}-{trg}"
    print(f"Processing dataset ({fname_base})...")

    # Deine tokenizers
    if tok_model == "wt":  # Word tokenizer
        src_tok = WordTokenizer(padding=False, truncation=False, max_length=5000, lang=src)
        trg_tok = WordTokenizer(padding=False, truncation=False, max_length=5000, lang=trg)
    elif tok_model == "hft":  # Huggingface tokenizer
        src_tok = LitTokenizer(padding=False, truncation=False, max_length=5000, lang=src)
        trg_tok = LitTokenizer(padding=False, truncation=False, max_length=5000, lang=trg)
    else:
        raise ValueError("Unknown tokenizer")

    # Create path
    savepath = os.path.join(dataset, DATASET_TOK_NAME, f"{tok_model}.{VOCAB_SIZE}")
    Path(savepath).mkdir(parents=True, exist_ok=True)

    # Training tokenizer
    print(f"Training vocabs...")
    src_tok.train_vocab(os.path.join(DATASETS_PATH, dataset, DATASET_CLEAN_NAME, f"train.{src}"), vocab_size=VOCAB_SIZE, min_frequency=2, lower=True)
    trg_tok.train_vocab(os.path.join(DATASETS_PATH, dataset, DATASET_CLEAN_NAME, f"train.{trg}"), vocab_size=VOCAB_SIZE, min_frequency=2, lower=True)

    # Sanity check
    # text_src = "Hello world!"
    # text_src_enc = src_tok.tokenizer.encode(text_src)
    # text_src_dec = src_tok.tokenizer.decode(text_src_enc.ids)
    # print(f"Source tokenizer")
    # print(f"\tRaw text: {text_src}")
    # print(f"\tEncoded text: {text_src_enc.tokens}")
    # print(f"\tDecoded text: {text_src_dec}")
    # print("")
    # #
    # trg_tok.train_vocab([os.path.join(DATASETS_PATH, dataset, DATASET_SPLITS_NAME, f"train.{trg}")], vocab_size=VOCAB_SIZE)
    # text_trg = "Hola mundo!"
    # text_trg_enc = trg_tok.tokenizer.encode(text_trg)
    # text_trg_dec = trg_tok.tokenizer.decode(text_trg_enc.ids)
    # print(f"Target tokenizer")
    # print(f"\tRaw text: {text_trg}")
    # print(f"\tEncoded text: {text_trg_enc.tokens}")
    # print(f"\tDecoded text: {text_trg_dec}")
    # print("")

    # Save tokenizers
    src_tok.save_vocab(savepath, prefix=f"tok.{src}")
    trg_tok.save_vocab(savepath, prefix=f"tok.{trg}")

    # Load tokenizer
    if tok_model == "wt":
        src_tok.load_vocab(os.path.join(savepath, f"tok.{src}-vocab.txt"))
        trg_tok.load_vocab(os.path.join(savepath, f"tok.{trg}-vocab.txt"))
    elif tok_model == "hft":
        src_tok.load_vocab(os.path.join(savepath, f"tok.{src}-vocab.json"),
                          os.path.join(savepath, f"tok.{src}-merges.txt"))
        trg_tok.load_vocab(os.path.join(savepath, f"tok.{trg}-vocab.json"),
                          os.path.join(savepath, f"tok.{trg}-merges.txt"))
    else:
        raise ValueError("Unknown tokenizer")

    asdas = 3
