import os
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from datasets import Dataset

from mt.common import LitTokenizer


def get_tokenizers(datapath, src, trg):
    # Define Tokenizers
    lt_src = LitTokenizer(padding=True, truncation=True, max_length=1024, lang=src)
    lt_trg = LitTokenizer(padding=True, truncation=True, max_length=1024, lang=trg)

    # Load vocab
    lt_src.load_vocab(os.path.join(datapath, f"tok.{src}-vocab.json"),
                      os.path.join(datapath, f"tok.{src}-merges.txt"))
    lt_trg.load_vocab(os.path.join(datapath, f"tok.{trg}-vocab.json"),
                      os.path.join(datapath, f"tok.{trg}-merges.txt"))

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
        data = {src: lines_src, trg: lines_trg}
        df = pd.DataFrame(data, columns=[src, trg])
        return df

    # Load datasets
    print(f"Reading files... ({datapath})")
    datasets = {}
    for s in splits:
        df = build_dataframe(s)
        datasets[s] = Dataset.from_pandas(df)

    return datasets


def build_dataloader(dataset, tok_src, tok_trg, batch_size=1, num_workers=0):
    # Pre-process datasets (lazy)
    ds = dataset.map(lambda x: encode(x, tok_src, tok_trg), batched=True)

    # Dataset formats
    ds.set_format(type='torch', columns=['src', 'trg'])

    # Dataset to Pytorch DataLoader
    ds_loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                                            collate_fn=lambda x: collate_fn(x, tok_src, tok_trg),
                                            shuffle=True, pin_memory=True)
    return ds_loader


def encode(examples, tok_src, tok_trg):
    # Encode strings
    _src_tokenized = tok_src.tokenizer.encode_batch(examples[tok_src.lang])
    _trg_tokenized = tok_trg.tokenizer.encode_batch(examples[tok_trg.lang])

    # Select features
    src_tokenized = [{'ids': x.ids, 'attention_mask': x.attention_mask} for x in _src_tokenized]
    trg_tokenized = []
    for x in _trg_tokenized:
        mask = x.attention_mask
        mask[-1] = 0  # "Remove" <eos> for translation
        # lengths = len(x.attention_mask)  # needed due to padded inputs and masks
        trg_tokenized.append({'ids': x.ids, 'attention_mask': mask})  # , 'lengths': lengths
    new_examples = {'src': src_tokenized, 'trg': trg_tokenized}
    return new_examples


def collate_fn(examples, tok_src, tok_trg):
    # Decompose examples
    _src = [x['src'] for x in examples]
    _trg = [x['trg'] for x in examples]

    # Processed examples
    src = tok_src.pad(_src, keys=['ids', 'attention_mask'])
    trg = tok_trg.pad(_trg, keys=['ids', 'attention_mask'])

    # Convert list to PyTorch tensor
    new_examples = [torch.stack(src['ids']), torch.stack(src['attention_mask']),
                    torch.stack(trg['ids']), torch.stack(trg['attention_mask'])]
    return new_examples
