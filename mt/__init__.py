import os

# Constants
VOCAB_SIZE = 16000

# Main folders
if os.getenv('LOCAL_GPU'):
    ROOT_PATH = "/home/salva/Documents/Programming/Datasets/scielo/"
    FASTBPE_PATH = "/home/salva/Documents/packages/fastBPE/fast"
elif os.getenv('REMOTE_GPU'):
    ROOT_PATH = "/home/scarrion/datasets/scielo/"
    FASTBPE_PATH = "/home/scarrion/packages/fastBPE/fast"
else:  # Default, local
    ROOT_PATH = "/home/salva/Documents/Programming/Datasets/scielo/"
    FASTBPE_PATH = "/home/salva/Documents/packages/fastBPE/fast"

BASE_PATH = os.path.join(ROOT_PATH, "constrained")  # Main folder
RAW_PATH = os.path.join(BASE_PATH, "raw")
DATASETS_PATH = os.path.join(BASE_PATH, "datasets")

# Subfolders (one per dataset)
DATASET_RAW_NAME = "raw"
DATASET_SPLITS_NAME = "splits"
DATASET_CLEAN_NAME = "clean"
DATASET_BPE_NAME = f"bpe.{VOCAB_SIZE}"
DATASET_TOK_NAME = "tok"
DATASET_EVAL_NAME = "eval"
DATASET_LOGS_NAME = "logs"
DATASET_CHECKPOINT_NAME = "checkpoints"

