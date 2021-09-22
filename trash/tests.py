import torch
import os

path = "/home/scarrion/datasets/scielo/constrained/datasets/bpe.32000/health_fairseq_large_vhealth_es-en/"
from fairseq.models.transformer import TransformerModel
model = TransformerModel.from_pretrained(os.path.join(path, "checkpoints"),
                                         checkpoint_file='checkpoint_best.pt',
                                         data_name_or_path=os.path.join(path, "data-bin"),
                                         bpe='fastbpe',
                                         bpe_codes=os.path.join(path, "tok/bpe.32000/codes.en")
                                         )
params = sum(p.numel() for p in model.parameters())
print(params)
asdsd = 3
