import os
import pandas as pd

import torch
from torch.utils.data import DataLoader

from datasets import Dataset

from trainer.tok.word_tokenizer import WordTokenizer
from trainer.tok.lit_tokenizer import LitTokenizer
from trainer.tok.fastbpe_tokenizer import FastBPETokenizer
from tqdm import tqdm


def get_tokenizers(datapath, src, trg, tok_model="fastbpe"):
    # Define Tokenizer
    if tok_model == "fastbpe":
        src_tok = FastBPETokenizer(padding=False, truncation=True, max_length=200, lang=src)
        trg_tok = FastBPETokenizer(padding=False, truncation=True, max_length=200, lang=trg)

        # Load vocab
        src_tok.load_vocab(os.path.join(datapath, f"codes.{src}"), os.path.join(datapath, f"vocab.{src}"))
        trg_tok.load_vocab(os.path.join(datapath, f"codes.{trg}"), os.path.join(datapath, f"vocab.{trg}"))

    elif tok_model == "wt":
        # Do not use padding here. Datasets are preprocessed before batching
        src_tok = WordTokenizer(padding=False, truncation=False, max_length=5000, lang=src)
        trg_tok = WordTokenizer(padding=False, truncation=False, max_length=5000, lang=trg)

        # Load vocab
        src_tok.load_vocab(os.path.join(datapath, f"tok.{src}-vocab.txt"))
        trg_tok.load_vocab(os.path.join(datapath, f"tok.{trg}-vocab.txt"))

    elif tok_model == "hft":
        # Do not use padding here. Datasets are preprocessed before batching
        src_tok = LitTokenizer(padding=False, truncation=False, max_length=5000, lang=src)
        trg_tok = LitTokenizer(padding=False, truncation=False, max_length=5000, lang=trg)

        # Load vocab
        src_tok.load_vocab(os.path.join(datapath, f"tok.{src}-vocab.json"), os.path.join(datapath, f"tok.{src}-merges.txt"))
        trg_tok.load_vocab(os.path.join(datapath, f"tok.{trg}-vocab.json"), os.path.join(datapath, f"tok.{trg}-merges.txt"))

    else:
        raise ValueError("Unknown tokenizer")

    return src_tok, trg_tok


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
        datasets[s] = build_dataframe(s)
        # df = build_dataframe(s)
        # datasets[s] = Dataset.from_pandas(df)

    return datasets


def build_dataloader(dataset, src_tok, trg_tok, tokenizer_class, batch_size=1, max_tokens=4000, num_workers=0, shuffle=True):
    # Pre-process datasets (lazy), encode=static method
    tokenizer_class.src_tok = src_tok
    tokenizer_class.trg_tok = trg_tok

    ds = dataset.map(tokenizer_class.encode, batched=True)
    ds.set_format(type='torch', columns=['src', 'trg'])

    # Dataset to Pytorch DataLoader
    ds_loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                                            collate_fn=lambda x: tokenizer_class.collate_fn(x, src_tok, trg_tok, max_tokens),
                                            shuffle=shuffle, pin_memory=True)
    return ds_loader


def print_translations(hypothesis, references, source=None, limit=None):
    print("")
    print("Translations: ")
    source = source if source else [None]*len(hypothesis)
    for i, (src, hyp, ref) in enumerate(zip(source, hypothesis, references)):
        print(f"Translation #{i+1}: ")
        if src:
            print("\t- Src: " + src)
        print("\t- Ref: " + ref)
        print("\t- Hyp: " + hyp)

        # Set limit
        if limit and i+1 >= limit:
            break


def generate_translations(model, trg_tok, data_loader, max_length, beam_width):
    y_pred = []
    y_true = []
    for batch in tqdm(data_loader, total=len(data_loader)):
        src, src_mask, trg, trg_mask = batch

        # Get indexes
        sos_idx = trg_tok.word2idx[trg_tok.SOS_WORD]
        eos_idx = trg_tok.word2idx[trg_tok.EOS_WORD]

        # Get output
        translations = model.translate_batch(src, src_mask, sos_idx, eos_idx, max_length=max_length, beam_width=beam_width)

        # Keep only best
        outputs_ids = [top_trans[0][0] for top_trans in translations]

        # Convert ids2words
        y_pred += trg_tok.decode(outputs_ids)
        y_true += trg_tok.decode(trg)

    return y_pred, y_true
