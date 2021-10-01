import torch
import os
from fairseq.models.transformer import TransformerModel
from fairseq.models.fconv import FConvModel
from fairseq.models.lstm import LSTMModel


models = [
"europarl_fairseq_lstm_es-en",
"europarl_fairseq_50k_lstm_es-en",

"europarl_fairseq_50k_conv_es-en",
"europarl_fairseq_conv_es-en",

# "europarl_fairseq_50k_transxs_es-en",
"europarl_fairseq_50k_es-en",
"europarl_fairseq_es-en",
]

summary = ""
for bpe_size in [64, 32000]:
    for fname in models:
        path = f"/home/scarrion/datasets/scielo/constrained/datasets/bpe.{bpe_size}/{fname}/"

        if "lstm" in path:
            architecture = "LSTM"
            model = LSTMModel.from_pretrained(os.path.join(path, "checkpoints"),
                                               checkpoint_file='checkpoint_best.pt',
                                               data_name_or_path=os.path.join(path, "data-bin"),
                                               bpe='fastbpe',
                                               bpe_codes=os.path.join(path, f"tok/bpe.{bpe_size}/codes.en")
                                               )
        elif "conv" in path:
            architecture = "CNN"
            model = FConvModel.from_pretrained(os.path.join(path, "checkpoints"),
                                                     checkpoint_file='checkpoint_best.pt',
                                                     data_name_or_path=os.path.join(path, "data-bin"),
                                                     bpe='fastbpe',
                                                     bpe_codes=os.path.join(path, f"tok/bpe.{bpe_size}/codes.en")
                                                     )
        else:
            architecture = "Transformer"
            model = TransformerModel.from_pretrained(os.path.join(path, "checkpoints"),
                                                     checkpoint_file='checkpoint_best.pt',
                                                     data_name_or_path=os.path.join(path, "data-bin"),
                                                     bpe='fastbpe',
                                                     bpe_codes=os.path.join(path, f"tok/bpe.{bpe_size}/codes.en")
                                                     )
        params = sum(p.numel() for p in model.parameters())
        line = f"BPE: {bpe_size} | Model: {architecture} | Params: {params} | Name: {fname}\n"
        print(line)
        summary += line
        asdsd = 3

print("*********************")
print("*********************")
print("*********************")
print(summary)
