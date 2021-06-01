import os
import numpy as np
import random
import time
import math
from pathlib import Path
from einops import rearrange
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchsummary import summary

import sacrebleu
from datasets import load_metric
from torchtext.data.metrics import bleu_score

from tqdm import tqdm

from mt import utils
from mt import helpers
from mt.dataloaders.datasets import TranslationDataset

from mt import DATASETS_PATH, DATASET_CLEAN_NAME, DATASET_CLEAN_SORTED_NAME, DATASET_TOK_NAME, DATASET_LOGS_NAME, DATASET_CHECKPOINT_NAME
from mt.trainer.models.transformer.transformer import Transformer
from mt.trainer.models.transformer.transformer_dyn import TransformerDyn

from torch.utils.data.sampler import SequentialSampler
from torchnlp.samplers import BucketBatchSampler
from mt.dataloaders.max_tokens_batch_sampler import MaxTokensBatchSampler

MODEL_NAME = "transformer"
WANDB_PROJECT = "nmt"  # Run "wandb login" in the terminal

MAX_EPOCHS = 50
LEARNING_RATE = 0.5e-3
BATCH_SIZE = 64 #int(32*1.5)
MAX_TOKENS = 1024 #int(4096*1.5)
WARMUP_UPDATES = 4000
PATIENCE = 10
ACC_GRADIENTS = 1  # Tricky. It can hurt the training.
WEIGHT_DECAY = 0.0001
CLIP_GRADIENTS = 1.0
MULTIGPU = False
DEVICE1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # torch.device("cpu") #
DEVICE2 = DEVICE1  #torch.device("cuda:1" if torch.cuda.is_available() else "cpu")  # torch.device("cpu") #
NUM_WORKERS = 0  # If is not zero, the debugger freezes!!!
TOK_MODEL = "bpe"
TOK_SIZE = 16000
TOK_FOLDER = f"{TOK_MODEL}.{TOK_SIZE}"
LOWERCASE = False
TRUNCATE = True
MAX_LENGTH_TRUNC = 2000
SAMPLER_NAME = "maxtokens" #"maxtokens"  # bucket # None
START_FROM_CHECKPOINT_MODEL1 = "transformer_health_best.pt"
START_FROM_CHECKPOINT_MODEL2 = "transformer_health_best.pt"
MODEL_INTERPOLATION = [0.5, 0.25, 0.75]
PRINT_TRANSLATIONS = True

print(f"Device #1: {DEVICE1}")
print(f"Device #2: {DEVICE2}")
print(f"CUDA devices count: {torch.cuda.device_count()}")

###########################################################################
###########################################################################

# Deterministic environment
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class CustomLoss:

    def __init__(self, alpha, loss1, loss2):
        self.loss1 = loss1
        self.loss2 = loss2
        self.alpha = alpha

    def custom_loss(self, output1, target1, output2, target2):
        l1 = self.loss1(output1, target1)
        l2 = self.loss2(output2, target2)
        return alpha * l1 + (1-alpha) * l2


def len_func(ds, i):
    return len(ds.datasets.iloc[i]["src"].split())


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


def freeze_layers(model):
    for param in model.parameters():
        param.requires_grad = False


def run_experiment(datapath, src, trg, alpha, domain=None, smart_batch=False):
    start_time = time.time()
    experiment_name = f"{domain}_lwf_a{alpha}_{src}-{trg}_small_1gpu"
    model_name = f"{MODEL_NAME}_{domain}_a{alpha}"

    ###########################################################################
    ###########################################################################

    wandb.init(project=WANDB_PROJECT, entity='salvacarrion', name=experiment_name, reinit=True)
    config = wandb.config
    config.model_name = MODEL_NAME
    config.domain = domain
    config.max_epochs = MAX_EPOCHS
    config.learning_rate = LEARNING_RATE
    config.batch_size = BATCH_SIZE
    config.max_tokens = MAX_TOKENS
    config.warmup_updates = WARMUP_UPDATES
    config.patience = PATIENCE
    config.acc_gradients = ACC_GRADIENTS
    config.weight_decay = WEIGHT_DECAY
    config.clip_gradients = CLIP_GRADIENTS
    config.multigpu = MULTIGPU
    config.device1 = str(DEVICE1)
    config.device2 = str(DEVICE2)
    config.num_workers = NUM_WORKERS
    config.tok_model = TOK_MODEL
    config.tok_size = TOK_SIZE
    config.tok_folder = TOK_FOLDER
    config.lowercase = LOWERCASE
    config.truncate = TRUNCATE
    config.max_length_truncate = MAX_LENGTH_TRUNC
    config.sampler_name = str(SAMPLER_NAME)
    config.start_from_checkpoint1 = START_FROM_CHECKPOINT_MODEL1
    config.start_from_checkpoint2 = START_FROM_CHECKPOINT_MODEL2
    config.model_interpolation = alpha
    print(config)

    ###########################################################################
    ###########################################################################

    # Load tokenizers
    src_tok, trg_tok = helpers.get_tokenizers(os.path.join(datapath, DATASET_TOK_NAME, TOK_FOLDER), src, trg,
                                              tok_model=TOK_MODEL, lower=LOWERCASE, truncation=TRUNCATE, max_length=MAX_LENGTH_TRUNC)

    # Load dataset
    datapath_clean = DATASET_CLEAN_SORTED_NAME if smart_batch else DATASET_CLEAN_NAME
    if TOK_MODEL == "bpe":  # Do not preprocess again when using bpe
        src_tok.apply_bpe = False
        trg_tok.apply_bpe = False
        datapath_clean = os.path.join(DATASET_TOK_NAME, TOK_FOLDER)

    train_ds = TranslationDataset(os.path.join(datapath, datapath_clean), src_tok, trg_tok, "train")
    val_ds_olddomain = TranslationDataset(os.path.join(os.path.join(DATASETS_PATH, "health_es-en"), datapath_clean), src_tok, trg_tok, "val")
    val_ds_newdomain = TranslationDataset(os.path.join(datapath, datapath_clean), src_tok, trg_tok, "val")

    # Build dataloaders
    kwargs_train = {}
    kwargs_val = {}
    if SAMPLER_NAME == "bucket":
        train_sampler = BucketBatchSampler(SequentialSampler(train_ds), batch_size=BATCH_SIZE, drop_last=False,
                                           sort_key=lambda i: len_func(train_ds, i))
        val_sampler_olddomain = BucketBatchSampler(SequentialSampler(val_ds_olddomain), batch_size=BATCH_SIZE, drop_last=False,
                                         sort_key=lambda i: len_func(val_ds_olddomain, i))
        val_sampler_newdomain = BucketBatchSampler(SequentialSampler(val_ds_newdomain), batch_size=BATCH_SIZE, drop_last=False,
                                         sort_key=lambda i: len_func(val_ds_newdomain, i))
    elif SAMPLER_NAME == "maxtokens":
        train_sampler = MaxTokensBatchSampler(SequentialSampler(train_ds), shuffle=True, batch_size=BATCH_SIZE,
                                              max_tokens=MAX_TOKENS, drop_last=False,
                                              sort_key=lambda i: len_func(train_ds, i))
        val_sampler_olddomain = MaxTokensBatchSampler(SequentialSampler(val_ds_olddomain), shuffle=False, batch_size=BATCH_SIZE,
                                             max_tokens=MAX_TOKENS, drop_last=False,
                                             sort_key=lambda i: len_func(val_ds_olddomain, i))
        val_sampler_newdomain = MaxTokensBatchSampler(SequentialSampler(val_ds_newdomain), shuffle=False, batch_size=BATCH_SIZE,
                                             max_tokens=MAX_TOKENS, drop_last=False,
                                             sort_key=lambda i: len_func(val_ds_newdomain, i))
    else:
        train_sampler = val_sampler_olddomain = val_sampler_newdomain = None
        kwargs_train = {"batch_size": BATCH_SIZE, "shuffle": True}
        kwargs_val = {"batch_size": BATCH_SIZE, "shuffle": False}

    # Define dataloader
    train_loader = DataLoader(train_ds, num_workers=NUM_WORKERS, collate_fn=lambda x: TranslationDataset.collate_fn(x, MAX_TOKENS), pin_memory=True, batch_sampler=train_sampler, **kwargs_train)
    val_loader_olddomain = DataLoader(val_ds_olddomain, num_workers=NUM_WORKERS, collate_fn=lambda x: TranslationDataset.collate_fn(x, MAX_TOKENS), pin_memory=True, batch_sampler=val_sampler_olddomain, **kwargs_val)
    val_loader_newdomain = DataLoader(val_ds_newdomain, num_workers=NUM_WORKERS, collate_fn=lambda x: TranslationDataset.collate_fn(x, MAX_TOKENS), pin_memory=True, batch_sampler=val_sampler_newdomain, **kwargs_val)

    # Instantiate model #1
    model1 = TransformerDyn(d_model=256,
                        enc_layers=3, dec_layers=3,
                        enc_heads=8, dec_heads=8,
                        enc_dff_dim=512, dec_dff_dim=512,
                        enc_dropout=0.1, dec_dropout=0.1,
                        max_src_len=2000, max_trg_len=2000,
                        src_tok=src_tok, trg_tok=trg_tok,
                        static_pos_emb=True).to(DEVICE1)
    print(f'The model #1 has {model1.count_parameters():,} trainable parameters')
    model1.apply(initialize_weights)

    # Load weights
    if START_FROM_CHECKPOINT_MODEL1:
        from_checkpoint_path = os.path.join(datapath, DATASET_CHECKPOINT_NAME, START_FROM_CHECKPOINT_MODEL1)
        print(f"(Model 1) Loading weights from: {from_checkpoint_path}")
        model1.load_state_dict(torch.load(from_checkpoint_path))

    # Process
    # train_sampler.shuffle = False
    # train_outputs = process_dataset(model1, train_loader, device=DEVICE1)
    # train_sampler.shuffle = True

    model2 = Transformer(d_model=256,
                        enc_layers=3, dec_layers=3,
                        enc_heads=8, dec_heads=8,
                        enc_dff_dim=512, dec_dff_dim=512,
                        enc_dropout=0.1, dec_dropout=0.1,
                        max_src_len=2000, max_trg_len=2000,
                        src_tok=src_tok, trg_tok=trg_tok,
                        static_pos_emb=True).to(DEVICE2)
    print(f'The model #2 has {model2.count_parameters():,} trainable parameters')
    model2.apply(initialize_weights)

    # [MODEL1] Freeze embedding layers and share parameters
    for param in model1.parameters():
        param.requires_grad = False
    model1.encoder_shared = model2.encoder
    model1.decoder_shared = model2.decoder

    # Load weights
    if START_FROM_CHECKPOINT_MODEL2:
        from_checkpoint_path = os.path.join(datapath, DATASET_CHECKPOINT_NAME, START_FROM_CHECKPOINT_MODEL2)
        print(f"(Model 2) Loading weights from: {from_checkpoint_path}")
        model2.load_state_dict(torch.load(from_checkpoint_path))

    optimizer = torch.optim.Adam(model2.parameters(), lr=LEARNING_RATE)
    cross_entropy_loss1 = nn.CrossEntropyLoss(ignore_index=trg_tok.word2idx[trg_tok.PAD_WORD])
    cross_entropy_loss2 = nn.CrossEntropyLoss(ignore_index=trg_tok.word2idx[trg_tok.PAD_WORD])
    criterion = CustomLoss(alpha, cross_entropy_loss1, cross_entropy_loss2)

    # Tensorboard (it needs some epochs to start working ~10-20)
    tb_writer = SummaryWriter(os.path.join(datapath, DATASET_LOGS_NAME, f"{model_name}"))
    wandb.watch(model2)

    # Train and validate model
    fit(model1, model2, optimizer, train_loader=train_loader, val_loader_olddomain=val_loader_olddomain, val_loader_newdomain=val_loader_newdomain,
        epochs=MAX_EPOCHS, criterion=criterion,
        checkpoint_path=os.path.join(datapath, DATASET_CHECKPOINT_NAME, model_name),
        tb_writer=tb_writer)

    print("************************************************************")
    epoch_hours, epoch_mins, epoch_secs = helpers.epoch_time(start_time, end_time=time.time())
    print(f'Time experiment: {epoch_hours}h {epoch_mins}m {epoch_secs}s')
    print("************************************************************")
    print("Done!")


def fit(model1, model2, optimizer, train_loader, val_loader_olddomain, val_loader_newdomain, epochs, criterion, checkpoint_path, tb_writer=None):
    if not checkpoint_path:
        print("[WARNING] Training without a checkpoint path. The model won't be saved.")

    # Evaluate
    val_loss_olddomain, translations_olddomain = evaluate(model2, val_loader_olddomain, criterion)
    val_loss_newdomain, translations_newdomain = evaluate(model2, val_loader_newdomain, criterion)

    # Log progress
    start_time = time.time()
    metrics_old = log_progress(-1, start_time, 1e1, val_loss_olddomain, tb_writer, translations_olddomain, print_translations=PRINT_TRANSLATIONS, prefix="old_domain")
    metrics_new = log_progress(-1, start_time, 1e1, val_loss_newdomain, tb_writer, translations_newdomain, print_translations=PRINT_TRANSLATIONS, prefix="new_domain")

    best_score = -1e9  # Loss 1e9; BLEU -1e9
    last_checkpoint = 0
    for epoch_i in range(epochs):
        start_time = time.time()

        # Train model
        tr_loss = train(model1, model2, optimizer, train_loader, criterion)

        # Evaluate
        val_loss_olddomain, translations_olddomain = evaluate(model2, val_loader_olddomain, criterion)
        val_loss_newdomain, translations_newdomain = evaluate(model2, val_loader_newdomain, criterion)

        # Log progress
        metrics_old = log_progress(epoch_i, start_time, tr_loss, val_loss_olddomain, tb_writer, translations_olddomain, print_translations=PRINT_TRANSLATIONS, prefix="old_domain")
        metrics_new = log_progress(epoch_i, start_time, tr_loss, val_loss_newdomain, tb_writer, translations_newdomain, print_translations=PRINT_TRANSLATIONS, prefix="new_domain")

        # Checkpoint
        new_best_score = save_checkpoint(model2, checkpoint_path, metrics_new, best_score)
        last_checkpoint = epoch_i if best_score != new_best_score else last_checkpoint
        best_score = new_best_score

        # Early stop
        if PATIENCE != -1 and (epoch_i - last_checkpoint) >= PATIENCE:
            print(f"************************************************************************")
            print(f"*** Early stop. Validation loss didn't improve for {PATIENCE} epochs ***")
            print(f"************************************************************************")
            break


def save_checkpoint(model, checkpoint_path, metrics, best_score):
    # Save checkpoint
    if checkpoint_path:
        # Save last
        torch.save(model.state_dict(), checkpoint_path + "_last.pt")

        # Save best BLEU
        score = metrics["val"]["bleu"]
        if score > best_score:  # Loss <; BLEU >
            best_score = score
            torch.save(model.state_dict(), checkpoint_path + "_best.pt")
            print("\t=> Checkpoint saved!")
    return best_score


def process_dataset(model, data_loader, device):
    outputs = []
    model.eval()
    for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
        # Get batch data
        src, src_mask, trg, trg_mask = [x.to(device) for x in batch]

        # Get output #1
        output, _ = model(src, src_mask, trg[:, :-1], trg_mask[:, :-1])
        outputs.append(output.detach().cpu())
    return outputs


def train(model1, model2, optimizer, data_loader, criterion):
    epoch_loss = 0.0

    model1.train()
    model2.train()
    optimizer.zero_grad()
    for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
        try:
            # Get batch data
            src1, src_mask1, trg1, trg_mask1 = [x.to(DEVICE1) for x in batch]
            src2, src_mask2, trg2, trg_mask2 = src1, src_mask1, trg1, trg_mask1  #[x.to(DEVICE2) for x in batch]
            batch_size, src_max_len, trg_max_len = src1.shape[0], src1.shape[1], trg1.shape[1]

            # Get output (target)
            with torch.no_grad():
                output1, _ = model1(src1, src_mask1, trg1[:, :-1], trg_mask1[:, :-1])
                trg0 = output1.argmax(2).contiguous().view(-1).long()

            # Get output #1 (shared)
            output1, _ = model1.forward_shared(src1, src_mask1, trg1[:, :-1], trg_mask1[:, :-1])
            output_dim1 = output1.shape[-1]
            output1 = output1.contiguous().view(-1, output_dim1)

            # Get output #2 (shared)
            output2, _ = model2(src2, src_mask2, trg2[:, :-1], trg_mask2[:, :-1])
            output_dim2 = output2.shape[-1]
            output2 = output2.contiguous().view(-1, output_dim2)
            trg2 = trg2[:, 1:].contiguous().view(-1).long()

            # Interpolation
            # output1 = output1.to(DEVICE2)

            # Compute loss
            loss = criterion.custom_loss(output1, trg0, output2, trg2)
            loss /= ACC_GRADIENTS  # Normalize loss
            loss.backward()

            # Track total loss
            epoch_loss += loss.item()

            # Accumulate gradients
            if (i+1) % ACC_GRADIENTS == 0 or (i+1) == len(data_loader):
                # Clip gradients
                if CLIP_GRADIENTS > 0:
                    torch.nn.utils.clip_grad_norm_(model2.parameters(), CLIP_GRADIENTS)
                optimizer.step()
                optimizer.zero_grad()
        except RuntimeError as e:
            print("ERROR BATCH: " + str(i+1))
            print(e)

    return epoch_loss / len(data_loader)


def evaluate(model, data_loader, criterion):
    epoch_loss = 0
    src_dec_all, hyp_dec_all, ref_dec_all = [], [], []

    model.eval()
    for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
        try:
            with torch.no_grad():
                # Get batch data
                src, src_mask, trg, trg_mask = [x.to(DEVICE2) for x in batch]

                # Get output
                # output, _ = model(src, trg[:, :-1])
                output, _ = model(src, src_mask, trg[:, :-1], trg_mask[:, :-1])
                _output = output.contiguous().view(-1, output.shape[-1])
                _trg = trg[:, 1:].contiguous().view(-1)

                # Compute loss
                loss = criterion.loss2(_output, _trg.long())
                epoch_loss += loss.item()

                # Generate translations (fast)
                hyp_dec_all += model.trg_tok.decode(output.argmax(2), remove_special_tokens=True)
                ref_dec_all += model.trg_tok.decode(trg, remove_special_tokens=True)
                src_dec_all += model.src_tok.decode(src, remove_special_tokens=True)
        except RuntimeError as e:
            print("ERROR BATCH: " + str(i+1))
            print(e)

    return epoch_loss / len(data_loader), (src_dec_all, hyp_dec_all, ref_dec_all)


def log_progress(epoch_i, start_time, tr_loss, val_loss, tb_writer=None, translations=None, print_translations=True, prefix=""):
    metrics = {
        "train": {
            "loss": tr_loss,
            "ppl": math.exp(tr_loss),
        },
        "val": {
            "loss": val_loss,
            "ppl": math.exp(val_loss),
        },
    }

    # Get additional metrics
    if translations:
        src_dec_all, hyp_dec_all, ref_dec_all = translations
        m_bleu_score = bleu_score([x.split(" ") for x in hyp_dec_all], [[x.split(" ")] for x in ref_dec_all])
        metrics["val"]["bleu"] = m_bleu_score*100

        # Print translations
        if print_translations:
            helpers.print_translations(hyp_dec_all, ref_dec_all, src_dec_all, limit=50)

    # Print stuff
    end_time = time.time()
    epoch_hours, epoch_mins, epoch_secs = helpers.epoch_time(start_time, end_time)
    print("------------------------------------------------------------")
    print(f'Epoch: {epoch_i + 1:02} | Time: {epoch_mins}m {epoch_secs}s | [DOMAIN: {prefix}]')
    print(f'\t[{prefix}]- Train Loss: {metrics["train"]["loss"]:.3f} | Train PPL: {metrics["train"]["ppl"]:.3f}')
    print(f'\t[{prefix}]- Val Loss: {metrics["val"]["loss"]:.3f} | Val PPL: {metrics["val"]["ppl"]:.3f} | Val BLEU: {metrics["val"]["bleu"]:.3f}')
    print("------------------------------------------------------------")

    # Tensorboard
    if tb_writer:
        for split in ["train", "val"]:
            for k, v in metrics[split].items():
                tb_writer.add_scalar(f'{prefix}_{split}_{k.lower()}', v, epoch_i+1)
                wandb.log({f'{prefix}_{split}_{k.lower()}': v})

    return metrics


if __name__ == "__main__":
    # Get all folders in the root path
    # datasets = [os.path.join(DATASETS_PATH, x) for x in ["health_es-en", "biological_es-en", "merged_es-en"]]
    # datasets = [os.path.join(DATASETS_PATH, x) for x in ["health_biological_es-en"]]
    datasets = [os.path.join(DATASETS_PATH, x) for x in ["health_biological_inter_es-en"]]
    for dataset in datasets:
        for alpha in MODEL_INTERPOLATION:
            domain, (src, trg) = utils.get_dataset_ids(dataset)
            fname_base = f"{domain}_{src}-{trg}"
            print(f"Training model ({fname_base})...")

            # Create paths
            Path(os.path.join(dataset, DATASET_LOGS_NAME)).mkdir(parents=True, exist_ok=True)
            Path(os.path.join(dataset, DATASET_CHECKPOINT_NAME)).mkdir(parents=True, exist_ok=True)

            # Train model
            run_experiment(dataset, src, trg, domain=domain, alpha=alpha)
