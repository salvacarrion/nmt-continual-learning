import os
import tqdm

SRC = "es"
TRG = "en"
MAX_TOKENS_SENTENCE = 1000
path_old = "/home/scarrion/datasets/scielo/constrained/datasets/bpe.64/europarl_fairseq_conv_es-en/tok/bpe.64_old"
path_new = "/home/scarrion/datasets/scielo/constrained/datasets/bpe.64/europarl_fairseq_conv_es-en/tok/bpe.64"
files = [(f"train.{SRC}", f"train.{TRG}"), (f"val.{SRC}", f"val.{TRG}"), (f"test.{SRC}", f"test.{TRG}")]

for file_src, file_trg in files:
    print(f"Cleaning files: {file_src} and {file_trg}...")
    # Read lines SRC
    with open(os.path.join(path_old, file_src), 'r') as f:
        file_src_lines = f.readlines()

    # Read lines TRG
    with open(os.path.join(path_old, file_trg), 'r') as f:
        file_trg_lines = f.readlines()

    # Remove lines
    print("Removing lines")
    new_src_lines = []
    new_trg_lines = []
    assert len(file_src_lines) == len(file_trg_lines)
    for src_line, trg_line in tqdm.tqdm(zip(file_src_lines, file_trg_lines), total=len(file_trg_lines)):
        if len(src_line.split(' ')) <= MAX_TOKENS_SENTENCE and len(trg_line.split(' ')) <= MAX_TOKENS_SENTENCE:
            new_src_lines.append(src_line)
            new_trg_lines.append(trg_line)

    assert len(new_src_lines) == len(new_trg_lines)
    print(f"Lines removed: {len(file_src_lines)-len(new_src_lines)}")
    del file_src_lines
    del file_trg_lines

    # Save split datasets
    with open(os.path.join(path_new, file_src), 'w', encoding='utf-8') as f:
        f.writelines([l.strip() + '\n' for l in new_src_lines])
    with open(os.path.join(path_new, file_trg), 'w', encoding='utf-8') as f:
        f.writelines([l.strip() + '\n' for l in new_trg_lines])
    print("Save files!")

print("Done!")
