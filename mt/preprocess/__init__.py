import os

# Constants
VOCAB_SIZE = 8000

# Main folders
if os.getenv('LOCAL_GPU'):
    ROOT_PATH = "/home/salvacarrion/Documents/Programming/Datasets/Scielo/"
    FASTBPE_PATH = "/home/salvacarrion/Documents/packages/fastBPE/fast"
elif os.getenv('REMOTE_GPU'):
    ROOT_PATH = "/home/scarrion/datasets/scielo/"
    FASTBPE_PATH = "/home/scarrion/packages/fastBPE/fast"
else:
    raise ValueError("Unknown environment")

BASE_PATH = os.path.join(ROOT_PATH, "constrained")  # Main folder
RAW_PATH = os.path.join(BASE_PATH, "raw")
DATASETS_PATH = os.path.join(BASE_PATH, "datasets")
LOGS_PATH = os.path.join(BASE_PATH, "logs")

# Subfolders (one per dataset)
DATASET_RAW_NAME = "raw"
DATASET_SPLITS_NAME = "splits"
DATASET_CLEAN_NAME = "clean"
DATASET_BPE_NAME = f"bpe.{VOCAB_SIZE}"
DATASET_TOK_NAME = "tok"

