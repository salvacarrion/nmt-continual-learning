import os
import torch
import math
import numpy as np

from tokenizers import Tokenizer, normalizers, pre_tokenizers
from tokenizers.normalizers import NFD, Lowercase, Strip, StripAccents
from tokenizers.pre_tokenizers import Punctuation, Whitespace, WhitespaceSplit

from tokenizers.models import WordPiece as tok_model
from tokenizers.trainers import WordPieceTrainer as tok_trainer
from tokenizers.decoders import WordPiece as tok_decoder
from tokenizers.processors import TemplateProcessing


def encode(examples, src_tok, trg_tok):
    # Encode strings
    _src_tokenized = src_tok.tokenizer.encode_batch(examples['src'])
    _trg_tokenized = trg_tok.tokenizer.encode_batch(examples['trg'])

    # Remove other params (there are problems with PyArrow)
    src_tokenized = [{'ids': x.ids, 'attention_mask': x.attention_mask} for x in _src_tokenized]
    trg_tokenized = []
    for x in _trg_tokenized:
        mask = x.attention_mask
        mask[-1] = 0  # "Remove" <eos> for translation
        # lengths = len(x.attention_mask)  # needed due to padded inputs and masks
        trg_tokenized.append({'ids': x.ids, 'attention_mask': mask})  # , 'lengths': lengths
    new_examples = {'src': src_tokenized, 'trg': trg_tokenized}
    return new_examples


def collate_fn(examples, src_tok, trg_tok, max_tokens):
    # Decompose examples
    _src = [x['src'] for x in examples]
    _trg = [x['trg'] for x in examples]

    # Processed examples
    src = src_tok.pad(_src, keys=['ids', 'attention_mask'])
    trg = trg_tok.pad(_trg, keys=['ids', 'attention_mask'])

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
    new_examples = [torch.stack(src['ids']), torch.stack(src['attention_mask']),
                    torch.stack(trg['ids']), torch.stack(trg['attention_mask'])]
    return new_examples


class LitTokenizer:

    def __init__(self, padding=False, truncation=False, max_length=None, lower=False, lang=None):
        super().__init__()
        self.UNK_WORD = '[UNK]'
        self.PAD_WORD = '[PAD]'
        self.MASK_WORD = '[MASK]'
        self.SOS_WORD = '[SOS]'
        self.EOS_WORD = '[EOS]'
        self.special_tokens = [self.UNK_WORD, self.PAD_WORD, self.MASK_WORD, self.SOS_WORD, self.EOS_WORD]

        # Define tokenizer
        self.tokenizer = None
        self.configure_tokenizers(padding, truncation, max_length, lower)

        # Other
        self.lang = lang

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def configure_tokenizers(self, padding, truncation, max_length, lower):
        # Settings
        pad_length = None
        if padding in {True, "longest"}:
            pass
        elif padding in {"max_length"}:
            pad_length = max_length
        elif padding in {False, "do_not_pad"}:
            pass
        else:
            raise ValueError("Unknown padding type")

        # SRC tokenizer
        tok_normalizers = [NFD(), Strip()]
        if lower:
            tok_normalizers += [Lowercase()]

        self.tokenizer = Tokenizer(tok_model())  # unk_token=... not working
        self.tokenizer.add_special_tokens(self.special_tokens)
        self.tokenizer.pre_tokenizer = pre_tokenizers.Sequence([WhitespaceSplit()])
        self.tokenizer.normalizer = normalizers.Sequence(tok_normalizers)  # StripAccents requires NFD
        self.tokenizer.decoder = tok_decoder()

        # Define template (Needed for the sos/eos tokens)
        basic_template = TemplateProcessing(
            single=f"{self.SOS_WORD} $A {self.EOS_WORD}",
            pair=f"{self.SOS_WORD} $A {self.EOS_WORD} {self.SOS_WORD} $B {self.EOS_WORD}",
            special_tokens=[(self.SOS_WORD, self.tokenizer.token_to_id(self.SOS_WORD)),
                            (self.EOS_WORD, self.tokenizer.token_to_id(self.EOS_WORD))],
        )
        self.tokenizer.post_processor = basic_template

        if padding:
            self.tokenizer.enable_padding(pad_id=self.tokenizer.token_to_id(self.PAD_WORD), pad_token=self.PAD_WORD, length=pad_length)
        if truncation:
            self.tokenizer.enable_truncation(max_length, stride=0, strategy='longest_first')

    def load_vocab(self, vocab, merges):
        vocab, merges = tok_model.read_file(vocab, merges)
        self.tokenizer.model = tok_model(vocab, merges)

    def train_vocab(self, files, vocab_size=32000, min_frequency=3):
        # Train trainer
        trainer = tok_trainer(vocab_size=vocab_size, min_frequency=min_frequency)
        self.tokenizer.train(files, trainer)

    def save_vocab(self, output_dir, prefix):
        self.tokenizer.model.save(output_dir, prefix)

    def pad(self, examples, keys=None):
        pad_idx = self.special_tokens.index(self.PAD_WORD)

        # Keys to modify
        if not keys:
            keys = list(examples[0].keys())

        d = {}
        for k in keys:
            # Collect same-type items (list of IDs, list of masks,...)
            d[k] = [x[k] for x in examples]

            # Get max length (value to pad)
            max_length = max([x.shape[-1] for x in d[k]])

            # Apply padding
            for i, x in enumerate(examples):
                unpadded_t = x[k]
                if k == "ids":
                    tmp = torch.full((max_length,), fill_value=pad_idx, device=unpadded_t.device)  # All padding
                elif k == "attention_mask":
                    tmp = torch.full((max_length,), fill_value=0, device=unpadded_t.device)  # No attention mask
                else:
                    raise TypeError("Unknown key")
                tmp[:unpadded_t.shape[-1]] = unpadded_t
                d[k][i] = tmp
        return d

    def encode(self, x):
        return self.tokenizer.encode(x)

    def decode(self, x):
        if isinstance(x, torch.Tensor):
            assert len(x.shape) == 2
            x = x.detach().cpu().numpy()
        return [self.tokenizer.decode(x_i) for x_i in x]
