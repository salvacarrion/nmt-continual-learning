from mt.old.tokenizers.tokenizer import LitTokenizer

# Create tokenizer
lt_src = LitTokenizer(padding=True, truncation=True, max_length=1024)
lt_trg = LitTokenizer(padding=True, truncation=True, max_length=1024)
