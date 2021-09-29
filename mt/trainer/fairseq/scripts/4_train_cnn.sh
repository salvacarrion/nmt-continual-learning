#!/bin/bash

# Define constants
BASE_PATH=$1
WANDB_PROJECT=$2

# Show constants
echo "Training model (large)... ****************"
echo "- Base path: "$BASE_PATH
echo "- W&B project name: "$WANDB_PROJECT
echo $(which fairseq-train)

# Train model
fairseq-train \
    $BASE_PATH/data-bin \
    --arch fconv \
    --optimizer adam \
    --criterion cross_entropy \
    --max-tokens 2048 \
    --max-epoch	50 \
    --seed 1234 \
    --clip-norm 1.0 \
    --patience 5 \
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
    --num-workers	$(nproc) \
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

