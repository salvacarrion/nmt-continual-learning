#!/bin/bash

# Define constants
EVAL_PATH=$1
MODEL_PATH=$2
OUTPUT_PATH=$3
SRC_LANG=$4
TRG_LANG=$5

# Show constants
echo "Evaluating model... ****************"
echo "- Evaluate path: "$EVAL_PATH
echo "- Model path: "$MODEL_PATH
echo "- Output path: "$OUTPUT_PATH
echo "- Source language: "$SRC_LANG
echo "- Target language: "$TRG_LANG

# Evaluate model
fairseq-generate \
    $EVAL_PATH \
    --source-lang $SRC_LANG --target-lang $TRG_LANG \
    --path $MODEL_PATH \
    --results-path $OUTPUT_PATH \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --num-workers $(nproc) \


#fairseq-generate data-bin/scielo_health_es_en/ --source-lang es --target-lang en --path checkpoints/transformer/checkpoint_best.pt --tokenizer moses --remove-bpe --beam 5 --scoring bleu
#fairseq-interactive data-bin/scielo_health_es_en/ --path checkpoints/transformer/checkpoint_best.pt --beam 5 --source-lang es --target-lang en --tokenizer moses --bpe fastbpe --bpe-codes codes
