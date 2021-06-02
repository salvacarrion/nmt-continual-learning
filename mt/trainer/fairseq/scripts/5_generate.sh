#!/bin/bash

# Define constants
SRC_LANG=$1
TRG_LANG=$2
BASE_PATH=$3
MODEL_PATH=$4
EVAL_PATH="${BASE_PATH}/eval"

# Show constants
echo "Evaluating model... ****************"
echo "- Source language: "$SRC_LANG
echo "- Target language: "$TRG_LANG
echo "- Model path: "$MODEL_PATH
echo "- Evaluate path: "$EVAL_PATH
echo "- Base path: "$BASE_PATH


# Create folder
mkdir -p $EVAL_PATH

# Evaluate model
fairseq-generate \
    $BASE_PATH/data-bin \
    --source-lang $SRC_LANG --target-lang $TRG_LANG \
    --path $MODEL_PATH/checkpoints/checkpoint_best.pt \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --results-path $EVAL_PATH \
     --num-workers $(nproc) \


#fairseq-generate data-bin/scielo_health_es_en/ --source-lang es --target-lang en --path checkpoints/transformer/checkpoint_best.pt --tokenizer moses --remove-bpe --beam 5 --scoring bleu
#fairseq-interactive data-bin/scielo_health_es_en/ --path checkpoints/transformer/checkpoint_best.pt --beam 5 --source-lang es --target-lang en --tokenizer moses --bpe fastbpe --bpe-codes codes
