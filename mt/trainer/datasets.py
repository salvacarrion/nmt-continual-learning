import os
import math
import torch
import numpy as np

from mt import helpers
from torch.utils.data import Dataset, DataLoader

src_pad_idx = 1
trg_pad_idx = 1


class TranslationDataset(Dataset):
    def __init__(self, path, src_tok, trg_tok, split):
        self.src_tok = src_tok
        self.trg_tok = trg_tok
        self.datasets = helpers.load_dataset(path, src_tok.lang, trg_tok.lang, splits=[split])[split]

        # Get pad indices
        src_pad_idx = self.src_tok.word2idx[self.src_tok.PAD_WORD]
        trg_pad_idx = self.trg_tok.word2idx[self.trg_tok.PAD_WORD]

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, idx):
        raw_sample = self.datasets.iloc[idx]
        src_enc = self.src_tok.encode_sample(raw_sample["src"], mask_eos=False)
        trg_enc = self.trg_tok.encode_sample(raw_sample["trg"], mask_eos=False)
        sample = {'src': src_enc["ids"], 'src_attention_mask': src_enc["attention_mask"],
                  'trg': trg_enc["ids"], 'trg_attention_mask': trg_enc["attention_mask"]}
        return sample

    @staticmethod
    def collate_fn(batch, max_tokens=4096//2):
        # Enable num_workers to make it fast

        # Build src pad tensor
        src_max_len = max([len(x["src"]) for x in batch])
        src = torch.full((len(batch), src_max_len), src_pad_idx, dtype=torch.int)
        src_attention_mask = torch.full((len(batch), src_max_len), 0, dtype=torch.bool)

        # Build trg pad tensor
        trg_max_len = max([len(x["trg"]) for x in batch])
        trg = torch.full((len(batch), trg_max_len), trg_pad_idx, dtype=torch.int)
        trg_attention_mask = torch.full((len(batch), trg_max_len), 0, dtype=torch.bool)

        # Add tensors
        for i, x in enumerate(batch):
            src[i, :len(x["src"])] = torch.tensor(x["src"], dtype=torch.int)
            src_attention_mask[i, :len(x["src_attention_mask"])] = torch.tensor(x["src_attention_mask"], dtype=torch.bool)
            trg[i, :len(x["trg"])] = torch.tensor(x["trg"], dtype=torch.int)
            trg_attention_mask[i, :len(x["trg_attention_mask"])] = torch.tensor(x["trg_attention_mask"], dtype=torch.bool)

        # Limit tokens
        batch_size, max_len = src.shape
        max_batch = math.floor(max_tokens / max_len)

        # Select indices
        if batch_size > max_batch:
            rnd_idxs = np.random.choice(np.arange(0, max_batch), size=max_batch, replace=False)
            src = src[rnd_idxs]
            src_attention_mask = src_attention_mask[rnd_idxs]
            trg = trg[rnd_idxs]
            trg_attention_mask = trg_attention_mask[rnd_idxs]

        return src, src_attention_mask, trg, trg_attention_mask

