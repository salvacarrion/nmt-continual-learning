import os
import pandas as pd

import torch
from torch.utils.data import DataLoader

from datasets import Dataset

from trainer.tok.lit_tokenizer import LitTokenizer
from trainer.tok.fastbpe_tokenizer import FastBPETokenizer


def get_tokenizers(datapath, src, trg, use_fastbpe):
    # Define Tokenizer
    if use_fastbpe:
        lt_src = FastBPETokenizer(padding=False, truncation=False, max_length=5000, lang=src)
        lt_trg = FastBPETokenizer(padding=False, truncation=False, max_length=5000, lang=trg)

        # Load vocab
        lt_src.load_vocab(os.path.join(datapath, f"codes.{src}"),
                          os.path.join(datapath, f"vocab.{src}"))
        lt_trg.load_vocab(os.path.join(datapath, f"codes.{trg}"),
                          os.path.join(datapath, f"vocab.{trg}"))
    else:
        # Do not use padding here. Datasets are preprocessed before batching
        lt_src = LitTokenizer(padding=False, truncation=False, max_length=5000, lang=src)
        lt_trg = LitTokenizer(padding=False, truncation=False, max_length=5000, lang=trg)

        # Load vocab
        lt_src.load_vocab(os.path.join(datapath, f"tok.{src}-vocab.json"),
                          os.path.join(datapath, f"tok.{src}-merges.txt"))
        lt_trg.load_vocab(os.path.join(datapath, f"tok.{trg}-vocab.json"),
                          os.path.join(datapath, f"tok.{trg}-merges.txt"))

    # # Sanity check
    # text_src = "Hola mundo!"
    # text_src_enc = lt_src.encode(text_src)
    # text_src_dec = lt_src.decode(text_src_enc.ids)
    # print(f"Source tokenizer")
    # print(f"\tRaw text: {text_src}")
    # print(f"\tEncoded text: {text_src_enc.tokens}")
    # print(f"\tDecoded text: {text_src_dec}")
    # print("")
    #
    # text_trg = "Hello world!"
    # text_trg_enc = lt_trg.encode(text_trg)
    # text_trg_dec = lt_trg.decode(text_trg_enc.ids)
    # print(f"Target tokenizer")
    # print(f"\tRaw text: {text_trg}")
    # print(f"\tEncoded text: {text_trg_enc.tokens}")
    # print(f"\tDecoded text: {text_trg_dec}")
    # print("")
    return lt_src, lt_trg


def load_dataset(datapath, src, trg, splits=None):
    if splits is None:
        splits = ["train", "val"]

    def load_from_text(split, lang):
        with open(os.path.join(datapath, f"{split}.{lang}"), 'r') as f:
            lines = [l.strip() for l in f.readlines()]
            return lines

    def build_dataframe(split):
        # Load files
        lines_src = load_from_text(split, src)
        lines_trg = load_from_text(split, trg)

        assert len(lines_src) == len(lines_trg)

        # Create pandas Dataframe
        data = {'src': lines_src, 'trg': lines_trg}
        df = pd.DataFrame(data, columns=['src', 'trg'])
        return df

    # Load datasets
    print(f"Reading files... ({datapath})")
    datasets = {}
    for s in splits:
        df = build_dataframe(s)
        datasets[s] = Dataset.from_pandas(df)

    return datasets


def build_dataloader(dataset, tok_src, tok_trg, apply_bpe=False, batch_size=1, max_tokens=4000, num_workers=0, shuffle=True):
    from trainer.tok.fastbpe_tokenizer import encode, collate_fn
    from trainer.tok import fastbpe_tokenizer

    # Pre-process datasets (lazy), encode=static method
    fastbpe_tokenizer.apply_bpe = apply_bpe
    fastbpe_tokenizer.tok_src = tok_src
    fastbpe_tokenizer.tok_trg = tok_trg
    ds = dataset.map(encode, batched=True)
    # ds = dataset.map(lambda x: encode(x, tok_src, tok_trg), batched=True)

    # Dataset formats
    ds.set_format(type='torch', columns=['src', 'trg'])

    # Dataset to Pytorch DataLoader
    ds_loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                                            collate_fn=lambda x: collate_fn(x, tok_src, tok_trg, max_tokens),
                                            shuffle=shuffle, pin_memory=True)
    return ds_loader


def print_translations(hypothesis, references):
    print("")
    print("Translations: ")
    for hyp, ref in zip(hypothesis, references):
        print("- Hyp: " + hyp)
        print("- Ref: " + ref)
        print("-------------------------")
