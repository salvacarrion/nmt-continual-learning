import os
import torch

from tokenizers import Tokenizer, normalizers, pre_tokenizers
from tokenizers.normalizers import NFD, Lowercase, Strip, StripAccents
from tokenizers.pre_tokenizers import Punctuation, Whitespace, WhitespaceSplit

from tokenizers.models import BPE as tok_model
from tokenizers.trainers import BpeTrainer as tok_trainer
from tokenizers.decoders import BPEDecoder as tok_decoder


class LitTokenizer:
    def __init__(self, padding=False, truncation=False, max_length=None):
        super().__init__()
        self.SOS_WORD = '[SOS]'
        self.EOS_WORD = '[EOS]'
        self.PAD_WORD = '[PAD]'
        self.UNK_WORD = '[UNK]'
        self.MASK_WORD = '[MASK]'
        self.special_tokens = [self.SOS_WORD, self.EOS_WORD, self.PAD_WORD, self.UNK_WORD, self.MASK_WORD]

        # Define tokenizer
        self.tokenizer = None
        self.configure_tokenizers(padding, truncation, max_length)

    def configure_tokenizers(self, padding, truncation, max_length):
        unk_idx = self.special_tokens.index(self.UNK_WORD)
        pad_idx = self.special_tokens.index(self.PAD_WORD)

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
        self.tokenizer = Tokenizer(tok_model())  # unk_token=... not working
        self.tokenizer.add_special_tokens(self.special_tokens)
        self.tokenizer.pre_tokenizer = pre_tokenizers.Sequence([Punctuation(), WhitespaceSplit()])
        self.tokenizer.normalizer = normalizers.Sequence([NFD(), Strip()])  # StripAccents requires NFD
        self.tokenizer.decoder = tok_decoder()
        if padding:
            self.tokenizer.enable_padding(pad_id=pad_idx, pad_token=self.PAD_WORD, length=pad_length)
        if truncation:
            self.tokenizer.enable_truncation(max_length, stride=0, strategy='longest_first')

    def load_vocab(self, vocab, merges):
        vocab, merges = tok_model.read_file(vocab, merges)
        self.tokenizer.model = tok_model(vocab, merges)

    def train_vocab(self, files, vocab_size=32000, min_frequency=3):
        # Train trainer
        trainer = tok_trainer(vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=self.special_tokens)
        self.tokenizer.train(files, trainer)

    def save_vocab(self, output_dir, prefix):
        self.tokenizer.model.save(output_dir, prefix)

    # def pad(self, examples, keys=None):
    #     pad_idx = self.special_tokens.index(self.PAD_WORD)
    #
    #     # Keys to modify
    #     if not keys:
    #         keys = list(examples[0].keys())
    #
    #     d = {}
    #     for k in keys:
    #         # Collect same-type items
    #         d[k] = [x[k] for x in examples]
    #
    #         # Get max length
    #         max_length = max([x.shape[-1] for x in d[k]])
    #
    #         # Apply padding
    #         for i, x in enumerate(examples):
    #             unpadded_t = x[k]
    #             if k == "ids":
    #                 tmp = torch.full((max_length,), fill_value=pad_idx, device=unpadded_t.device)  # All padding
    #             elif k == "attention_mask":
    #                 tmp = torch.full((max_length,), fill_value=0, device=unpadded_t.device)  # No attention mask
    #             else:
    #                 raise TypeError("Unknown key")
    #             tmp[:unpadded_t.shape[-1]] = unpadded_t
    #             d[k][i] = tmp
    #     return d