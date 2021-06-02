#!/bin/bash

# Define constants
BASE_PATH=$1

# Show constants
echo "Training model... ****************"
echo "- Base path: "$BASE_PATH

# Train model
fairseq-train \
    $BASE_PATH/data-bin \
    --arch transformer \
    --optimizer adam \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --num-workers	$(nproc) \
    --max-epoch	50 \
    --seed 1 \
    --save-dir $BASE_PATH/checkpoints \
    --log-format simple \
    --no-epoch-checkpoints \
    --tensorboard-logdir $BASE_PATH/logdir \
    --update-freq 8 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --clip-norm 1.0 \
    --lr 1e-3  \
    --lr-scheduler reduce_lr_on_plateau  \
    --warmup-updates 4000 \
    --dropout 0.1 --weight-decay 0.0001 \
    --patience 10 \
    --restore-file $BASE_PATH/checkpoints/health_checkpoint_best.pt \
    --reset-dataloader \
    --reset-lr-scheduler \
    --reset-meters \
    --reset-optimizer \
    --wandb-project "mltests-constrained"

echo "##########################################"
echo "Training finished!"
echo "##########################################"
exit 1
