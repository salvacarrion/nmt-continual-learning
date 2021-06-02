#!/bin/bash

# Define constants
SOURCE_DATASET=$1
VOCAB_PATH=$2
OUTPUT_PATH=$3
TOK_FOLDER=$4
SRC_LANG=$5
TRG_LANG=$6

# Show constants
echo "Preprocessing files for Fairseq... ****************"
echo "- Source dataset: "$SOURCE_DATASET
echo "- Vocab path: "$VOCAB_PATH
echo "- Output path: "$OUTPUT_PATH
echo "- Tok folder: "$TOK_FOLDER
echo "- Source language: "$SRC_LANG
echo "- Target language: "$TRG_LANG

# Preprocess files
fairseq-preprocess \
    --source-lang $SRC_LANG --target-lang $TRG_LANG \
    --trainpref $SOURCE_DATASET/tok/$TOK_FOLDER/train \
    --validpref $SOURCE_DATASET/tok/$TOK_FOLDER/val \
    --testpref $SOURCE_DATASET/tok/$TOK_FOLDER/test \
    --destdir $OUTPUT_PATH/data-bin \
    --srcdict	$VOCAB_PATH/tok/$TOK_FOLDER/vocab.${SRC_LANG} \
    --tgtdict	$VOCAB_PATH/tok/$TOK_FOLDER/vocab.${TRG_LANG} \
    --workers	$(nproc) \

