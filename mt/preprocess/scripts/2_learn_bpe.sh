#!/bin/bash

# Define constants
VOCAB_SIZE=$1
SRC_LANG=$2
TRG_LANG=$3
BASE_PATH=$4
SAVE_PATH=$5
FASTBPE_PATH=$6

# Fast BPE: https://github.com/glample/fastBPE

# Show constants
echo "Applying BPE... ****************"
echo "- Source language: "$SRC_LANG
echo "- Target language: "$TRG_LANG
echo "- Base path: "$BASE_PATH
echo "- Save path: "$SAVE_PATH

# Create folder
mkdir -p $SAVE_PATH

# Learn codes
$FASTBPE_PATH learnbpe $VOCAB_SIZE $BASE_PATH/clean/train.$SRC_LANG > $SAVE_PATH/codes.$SRC_LANG
$FASTBPE_PATH learnbpe $VOCAB_SIZE $BASE_PATH/clean/train.$TRG_LANG > $SAVE_PATH/codes.$TRG_LANG

