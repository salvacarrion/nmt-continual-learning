import os
import pandas as pd

from datasets import Dataset

from mt.common import LitTokenizer


def get_tokenizers(src, trg, datapath):
    # Define Tokenizers
    lt_src = LitTokenizer(padding=True, truncation=True, max_length=1024)
    lt_trg = LitTokenizer(padding=True, truncation=True, max_length=1024)

    # Load vocab
    lt_src.load_vocab(os.path.join(datapath, f"{src}-vocab.json"),
                      os.path.join(datapath, f"{src}-merges.txt"))
    lt_trg.load_vocab(os.path.join(datapath, f"{trg}-vocab.json"),
                      os.path.join(datapath, f"{trg}-merges.txt"))

    return lt_src, lt_trg


def load_dataset(datapath, src, trg, splits=None):
    if splits is None:
        splits = ["train", "val"]

    def load_from_text(split, lang):
        with open(os.path.join(datapath, f"{split}.{lang}"), 'r') as f:
            lines = f.readlines()
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
        datasets[s] = build_dataframe(s)

    return datasets

#
#  # Tokenize
# def encode(examples):
#     # Encode strings
#     _src_tokenized = lt_src.tokenizer.encode_batch(examples[SRC_LANG])
#     _trg_tokenized = lt_trg.tokenizer.encode_batch(examples[TRG_LANG])
#
#     # Select features
#     src_tokenized = [{'ids': x.ids, 'attention_mask': x.attention_mask} for x in _src_tokenized]
#     trg_tokenized = []
#     for x in _trg_tokenized:
#         mask = x.attention_mask
#         mask[-1] = 0  # "Remove" <eos> for translation
#         # lengths = len(x.attention_mask)  # needed due to padded inputs and masks
#         trg_tokenized.append({'ids': x.ids, 'attention_mask': mask})  # , 'lengths': lengths
#     new_examples = {'src': src_tokenized, 'trg': trg_tokenized}
#     return new_examples
#
#
# def collate_fn(examples):
#     # Decompose examples
#     _src = [x['src'] for x in examples]
#     _trg = [x['trg'] for x in examples]
#
#     # Processed examples
#     src = lt_trg.tokenizer.pad(_src, keys=['ids', 'attention_mask'])
#     trg = lt_trg.tokenizer.pad(_trg, keys=['ids', 'attention_mask'])
#
#     # Convert list to PyTorch tensor
#     new_examples = [torch.stack(src['ids']), torch.stack(src['attention_mask']),
#                     torch.stack(trg['ids']), torch.stack(trg['attention_mask'])]
#     return new_examples