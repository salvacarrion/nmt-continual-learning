#!/bin/bash

# Define constants
BASE_PATH=$1
WANDB_PROJECT=$2

# Show constants
echo "Training model (small)... ****************"
echo "- Base path: "$BASE_PATH
echo "- W&B project name: "$WANDB_PROJECT
echo $(which fairseq-train)

# Train model
fairseq-train \
    $BASE_PATH/data-bin \
    --arch transformer \
    --encoder-embed-dim 256 \
    --decoder-embed-dim 256 \
    --encoder-layers 3 \
    --decoder-layers 3 \
    --encoder-attention-heads	8 \
    --decoder-attention-heads	8 \
    --encoder-ffn-embed-dim	512 \
    --decoder-ffn-embed-dim	512 \
    --dropout	0.1 \
    --criterion cross_entropy \
    --max-tokens 4096 \
    --seed 1234 \
    --patience 10 \
    --max-epoch	75 \
    --lr 0.25 \
    --optimizer nag --clip-norm 0.1 \
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
#    --skip-invalid-size-inputs-valid-test
#    --restore-file $BASE_PATH/checkpoints/health_checkpoint_best.pt \
#    --reset-dataloader \
#    --reset-lr-scheduler \
#    --reset-meters \
#    --reset-optimizer \
#    --warmup-updates 4000 \
#    --update-freq 8 \
#    --lr-scheduler reduce_lr_on_plateau  \

echo "##########################################"
echo "Training finished!"
echo "##########################################"

