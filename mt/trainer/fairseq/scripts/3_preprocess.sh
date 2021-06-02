#!/bin/bash

# Define constants
BASE_PATH=$1
TOK_FOLDER=$2
SRC_LANG=$3
TRG_LANG=$4

# Show constants
echo "Preprocessing files for Fairseq... ****************"
echo "- Base path: "$BASE_PATH
echo "- Tok folder: "$TOK_FOLDER
echo "- Source language: "$SRC_LANG
echo "- Target language: "$TRG_LANG

# Preprocess files
fairseq-preprocess \
    --source-lang $SRC_LANG --target-lang $TRG_LANG \
    --trainpref $BASE_PATH/tok/$TOK_FOLDER/train \
    --validpref $BASE_PATH/tok/$TOK_FOLDER/val \
    --testpref $BASE_PATH/tok/$TOK_FOLDER/test \
    --destdir $BASE_PATH/data-bin \
    --workers	$(nproc) \
    --srcdict	$BASE_PATH/tok/$TOK_FOLDER/vocab.${SRC_LANG} \
    --tgtdict	$BASE_PATH/tok/$TOK_FOLDER/vocab.${TRG_LANG} \

