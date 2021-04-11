import os
import torch
import math
import numpy as np

import fastBPE

# def encode(examples, tok_src, tok_trg, apply_bpe=False):
apply_bpe = None
tok_src = None
tok_trg = None


def encode(examples):
    # Apply BPE if needed
    src_tokenized, trg_tokenized = examples['src'], examples['trg']
    if apply_bpe:
        src_tokenized = tok_src.apply_bpe(examples['src'])
        trg_tokenized = tok_trg.apply_bpe(examples['trg'])

    # Encode samples
    src_tokenized = [tok_src.encode_sample(x, mask_eos=False) for x in src_tokenized]
    trg_tokenized = [tok_trg.encode_sample(x, mask_eos=True) for x in trg_tokenized]

    return {'src': src_tokenized, 'trg': trg_tokenized}


def collate_fn(examples, tok_src, tok_trg, max_tokens):
    # Decompose examples
    _src = [x['src'] for x in examples]
    _trg = [x['trg'] for x in examples]

    # Processed examples
    src = tok_src.pad(_src, keys=['ids', 'attention_mask'])
    trg = tok_trg.pad(_trg, keys=['ids', 'attention_mask'])

    # Limit tokens
    batch_size, max_len = len(src['ids']), len(src['ids'][0])
    max_batch = math.floor(max_tokens / max_len)

    # Select indices
    if max_batch < batch_size:
        rnd_idxs = np.random.choice(np.arange(0, max_batch), size=max_batch, replace=False)
        src['ids'] = [src['ids'][i] for i in rnd_idxs]
        src['attention_mask'] = [src['attention_mask'][i] for i in rnd_idxs]
        trg['ids'] = [trg['ids'][i] for i in rnd_idxs]
        trg['attention_mask'] = [trg['attention_mask'][i] for i in rnd_idxs]

    # Convert list to PyTorch tensor
    new_examples = [torch.stack(src['ids']).type(torch.int), torch.stack(src['attention_mask']).type(torch.bool),
                    torch.stack(trg['ids']).type(torch.int), torch.stack(trg['attention_mask']).type(torch.bool)]
    return new_examples


class FastBPETokenizer:

    def __init__(self, padding=False, truncation=False, max_length=None, lang=None):
        super().__init__()
        self.UNK_WORD = '[UNK]'
        self.PAD_WORD = '[PAD]'
        self.MASK_WORD = '[MASK]'
        self.SOS_WORD = '[SOS]'
        self.EOS_WORD = '[EOS]'
        self.special_tokens = [self.UNK_WORD, self.PAD_WORD, self.MASK_WORD, self.SOS_WORD, self.EOS_WORD]
        self._special_tokens_set = set(self.special_tokens)
        self._special_tokens_ids_set = set(range(len(self.special_tokens)))

        # Define tokenizer
        self.tokenizer = None
        self.word2idx = {}
        self.idx2word = {}

        # Other
        self.lang = lang

    def get_vocab_size(self):
        return len(self.word2idx)

    def load_vocab(self, codes_path, vocab_path):
        self.tokenizer = fastBPE.fastBPE(codes_path, vocab_path)
        with open(vocab_path, 'r') as f:
            vocabs = [l.split(' ')[0].strip() for l in f.readlines()]
            for idx, word in enumerate(self.special_tokens + vocabs, 0):
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def pad(self, examples, keys):
        padded = {}
        for k in keys:
            # Collect same-type items (list of IDs, list of masks,...)
            padded[k] = [x[k] for x in examples]

            # Get max length (value to pad)
            max_length = max([x.shape[-1] for x in padded[k]])

            # Apply padding
            for i, x in enumerate(examples):
                unpadded_t = x[k]
                if k == "ids":
                    tmp = torch.full((max_length,), fill_value=self.word2idx[self.PAD_WORD], device=unpadded_t.device)  # All padding
                elif k == "attention_mask":
                    tmp = torch.full((max_length,), fill_value=0, device=unpadded_t.device)  # No attention mask
                else:
                    raise TypeError("Unknown key")
                tmp[:unpadded_t.shape[-1]] = unpadded_t
                padded[k][i] = tmp
        return padded

    def apply_bpe(self, x):
        if isinstance(x, str):
            x =[x]
        return self.tokenizer.apply(x)

    def decode_bpe(self, x):
        if isinstance(x, str):
            x =[x]
        return [x_i.replace("@@ ", "").strip() for x_i in x]

    def decode(self, x, return_str=True, decode_bpe=True, remove_special_tokens=True):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()

        # Convert ids to words
        if remove_special_tokens:
            sentences = [[self.idx2word[idx] for idx in x_i if idx not in self._special_tokens_ids_set] for x_i in x]
        else:
            sentences = [[self.idx2word[idx] for idx in x_i] for x_i in x]

        # Return sentences as strings
        if return_str:
            sentences = [" ".join(sent) for sent in sentences]

        # Decode bpe
        if decode_bpe:
            sentences = self.decode_bpe(sentences)

        return sentences

    def decode_with_mask(self, x, mask):
        return [[(ii, jj) for ii, jj in zip(i, j)] for i, j in zip(self.decode(x, return_str=False, decode_bpe=False, remove_special_tokens=False), mask.cpu().numpy())]

    def encode_sample(self, x, mask_eos=False):
        tokens = [self.SOS_WORD] + [w if w in self.word2idx else self.UNK_WORD for w in x.split(' ')] + [self.EOS_WORD]
        ids = [self.word2idx[w] for w in tokens]
        attention_mask = [1]*len(ids)

        # Mask EOS (The decoder cannot know above the EOS)
        if mask_eos:
            attention_mask[-1] = 0

        return {"ids": ids, "attention_mask": attention_mask}

