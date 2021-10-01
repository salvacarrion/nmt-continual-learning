#!/bin/bash

# Define constants
BASE_PATH=$1
WANDB_PROJECT=$2

# Show constants
echo "Training model (large)... ****************"
echo "- Base path: "$BASE_PATH
echo "- W&B project name: "$WANDB_PROJECT
echo $(which fairseq-train)

# Without NAG is does not converge ¿?
#CUDA_VISIBLE_DEVICES=0
fairseq-train \
    $BASE_PATH/data-bin \
    --arch lstm_wiseman_iwslt_de_en \
    --encoder-embed-dim 256 \
    --encoder-hidden-size 256 \
    --decoder-embed-dim 256 \
    --decoder-hidden-size 256 \
    --decoder-out-embed-dim 256 \
    --encoder-layers 3 \
    --decoder-layers 3 \
    --encoder-bidirectional \
    --criterion cross_entropy \
    --lr 0.25 \
    --optimizer nag --clip-norm 0.1 \
    --max-tokens 4096 \
    --seed 1234 \
    --patience 10 \
    --max-epoch	75 \
    --save-dir $BASE_PATH/checkpoints \
    --log-format simple \
    --no-epoch-checkpoints \
    --tensorboard-logdir $BASE_PATH/logs \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --task translation \
    --num-workers $(nproc) \
    --wandb-project $WANDB_PROJECT \
#    --ddp-backend=no_c10d \
#    --ddp-backend=legacy_ddp \
#    --ddp-backend=legacy_ddp \
#    --restore-file $BASE_PATH/checkpoints/health_checkpoint_best.pt \
#    --reset-dataloader \
#    --reset-lr-scheduler \
#    --reset-meters \
#    --reset-optimizer \
#    --update-freq 8 \
#    --lr-scheduler reduce_lr_on_plateau  \

echo "##########################################"
echo "Training finished!"
echo "##########################################"

