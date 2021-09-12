import os
import random
import re
from pathlib import Path
from torchtext.datasets import IWSLT2016, IWSLT2017
import unicodedata

# Vars
SRC_LANG = 'de'
TRG_LANG = 'en'
shuffle = True

# Create savepath
dataset_name = "IWSLT2016"
savepath = os.path.join(".", dataset_name, "splits")
Path(savepath).mkdir(parents=True, exist_ok=True)


def valid_line(line, min_length=1, max_length=2000, max_diff=250):
    # Min length
    if len(line[0]) < min_length or len(line[1]) < min_length:
        return False

    # Max length
    if len(line[0]) > max_length or len(line[1]) > max_length:
        return False

    # Max diff
    if abs(len(line[0])-len(line[1])) > max_diff:
        return False

    return True


def preprocess_text(text):
    try:
        # Remove repeated whitespaces "   " => " "
        p_whitespace = re.compile(" +")
        text = p_whitespace.sub(' ', text)

        # Normalization Form Compatibility Composition
        text = unicodedata.normalize("NFKC", text)  # Dangerous: https://www.gushiciku.cn/pl/pl4g

        # Strip whitespace
        text = text.strip()
    except TypeError as e:
        # print(f"=> Error preprocessing: '{text}'")
        text = ""
    return text

# Download dataset
train_iter, valid_iter, test_iter = IWSLT2016(language_pair=('de', 'en'))

# Convert to tuple and remove empty lines
ds = []
for lines in [valid_iter, test_iter]:
    lines_count = len(lines)
    lines = [(preprocess_text(t[0]), preprocess_text(t[1])) for t in lines]  # Preprocess
    lines = [t for t in lines if valid_line(t)]  # Remove empty lines
    lines_count2 = len(lines)
    print(f"Lines removed: {lines_count - lines_count2}")

    # Shuffle lines
    random.shuffle(lines) if shuffle else None

    # Add sets
    ds.append(lines)

# Unpack
train_src, train_trg = zip(*ds[0])
val_src, val_trg = zip(*ds[1])
test_src, test_trg = zip(*ds[2])

# Save split datasets
with open(os.path.join(savepath, f"train.{SRC_LANG}"), 'w', encoding='utf-8') as f:
    f.writelines([l + '\n' for l in train_src])
with open(os.path.join(savepath, f"train.{TRG_LANG}"), 'w', encoding='utf-8') as f:
    f.writelines([l + '\n' for l in train_trg])
with open(os.path.join(savepath, f"val.{SRC_LANG}"), 'w', encoding='utf-8') as f:
    f.writelines([l + '\n' for l in val_src])
with open(os.path.join(savepath, f"val.{TRG_LANG}"), 'w', encoding='utf-8') as f:
    f.writelines([l + '\n' for l in val_trg])
with open(os.path.join(savepath, f"test.{SRC_LANG}"), 'w', encoding='utf-8') as f:
    f.writelines([l + '\n' for l in test_src])
with open(os.path.join(savepath, f"test.{TRG_LANG}"), 'w', encoding='utf-8') as f:
    f.writelines([l + '\n' for l in test_trg])