import os
from pathlib import Path
import subprocess

from mt import DATASETS_PATH, DATASET_TOK_NAME, DATASET_BPE_NAME, FASTBPE_PATH
from mt import utils

VOCAB_SIZE = 64
TOK_FOLDER = f"{DATASET_BPE_NAME}.{VOCAB_SIZE}"
SAVE_VOCABS = True

# Get all folders in the root path
datasets = [os.path.join(DATASETS_PATH, TOK_FOLDER, x) for x in [
    "europarl_fairseq_es-en",

    # "europarl_fairseq_50k_de-en",
    # "europarl_fairseq_50k_es-en",
    # "europarl_fairseq_50k_cs-en",
    # "europarl_fairseq_cs-en",
    # "europarl_fairseq_100k_cs-en",
    # "europarl_fairseq_100k_de-en",
    # "europarl_fairseq_de-en",
    # "europarl_fairseq_fr-en",
    # "europarl_fairseq_cs-en",
    # "commoncrawl_es-en",
    # "commoncrawl_100k_es-en",
    # "newscommentaryv14_35k_es-en",
    # "newscommentaryv14_es-en",
    # "europarl_fairseq_100k_es-en",
    # "health_fairseq_vhealth_unconstrained_es-en",
    # "iwlst2016_de-en",
    #"multi30k_de-en",

    # "health_fairseq_vhealth_es-en",
    # "health_fairseq_vbiological_es-en",
    # "health_fairseq_vmerged_es-en",
    #
    # "biological_fairseq_vhealth_es-en",
    # "biological_fairseq_vbiological_es-en",
    # "biological_fairseq_vmerged_es-en",
    #
    # "merged_fairseq_vhealth_es-en",
    # "merged_fairseq_vbiological_es-en",
    # "merged_fairseq_vmerged_es-en",

    # "health_biological_fairseq_vbiological_es-en",
    # "health_biological_fairseq_vhealth_es-en",
    # "health_biological_fairseq_vmerged_es-en",
]]

# datasets = [os.path.join(DATASETS_PATH, "multi30k_de-en")]
for dataset in datasets:
    domain, (src, trg) = utils.get_dataset_ids(dataset)
    fname_base = f"{domain}_{src}-{trg}"
    print(f"Processing dataset ({fname_base})...")

    # Create path
    savepath = os.path.join(dataset, DATASET_TOK_NAME, TOK_FOLDER)
    Path(savepath).mkdir(parents=True, exist_ok=True)

    # Learn and apply BPE
    subprocess.call(['sh', './scripts/2_learn_bpe.sh', str(VOCAB_SIZE), src, trg, dataset, savepath, FASTBPE_PATH])
    subprocess.call(['sh', './scripts/2_apply_bpe.sh', str(VOCAB_SIZE), src, trg, dataset, savepath, FASTBPE_PATH, "true" if SAVE_VOCABS else "false"])
