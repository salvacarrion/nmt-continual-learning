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
    --arch transformer \
    --optimizer adam \
    --activation-fn relu \
    --encoder-embed-dim 512 \
    --decoder-embed-dim 512 \
    --encoder-layers 6 \
    --decoder-layers 6 \
    --encoder-attention-heads	8 \
    --decoder-attention-heads	8 \
    --encoder-ffn-embed-dim	2048 \
    --decoder-ffn-embed-dim	2048 \
    --dropout	0.1 \
    --criterion cross_entropy \
    --max-tokens 4096 \
    --max-epoch	50 \
    --seed 1234 \
    --clip-norm 1.0 \
    --lr 0.5e-3 \
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
    --wandb-project $WANDB_PROJECT \
    --num-workers	$(nproc) \
#    --restore-file $BASE_PATH/checkpoints/health_checkpoint_best.pt \
#    --reset-dataloader \
#    --reset-lr-scheduler \
#    --reset-meters \
#    --reset-optimizer \
#    --warmup-updates 4000\
#    --update-freq 8 \
#    --lr-scheduler reduce_lr_on_plateau  \

echo "##########################################"
echo "Training finished!"
echo "##########################################"

