#!/bin/sh

# Define constants
VOCAB_SIZE=$1
SRC_LANG=$2
TRG_LANG=$3
BASE_PATH=$4
FASTBPE_PATH=$5

# Fast BPE: https://github.com/glample/fastBPE

# Show constants
echo "Applying BPE... ****************"
echo "- Source language: "$SRC_LANG
echo "- Target language: "$TRG_LANG
echo "- Base path: "$BASE_PATH

# Create folder
mkdir -p $BASE_PATH/bpe/

# Apply BPE ****
# Source
$FASTBPE_PATH applybpe $BASE_PATH/bpe/train.$SRC_LANG $BASE_PATH/clean/train.$SRC_LANG $BASE_PATH/bpe/codes.$SRC_LANG
$FASTBPE_PATH applybpe $BASE_PATH/bpe/val.$SRC_LANG $BASE_PATH/clean/val.$SRC_LANG $BASE_PATH/bpe/codes.$SRC_LANG
$FASTBPE_PATH applybpe $BASE_PATH/bpe/test.$SRC_LANG $BASE_PATH/clean/test.$SRC_LANG $BASE_PATH/bpe/codes.$SRC_LANG

# Target
$FASTBPE_PATH applybpe $BASE_PATH/bpe/train.$TRG_LANG $BASE_PATH/clean/train.$TRG_LANG $BASE_PATH/bpe/codes.$TRG_LANG
$FASTBPE_PATH applybpe $BASE_PATH/bpe/val.$TRG_LANG $BASE_PATH/clean/val.$TRG_LANG $BASE_PATH/bpe/codes.$TRG_LANG
$FASTBPE_PATH applybpe $BASE_PATH/bpe/test.$TRG_LANG $BASE_PATH/clean/test.$TRG_LANG $BASE_PATH/bpe/codes.$TRG_LANG

# Save vocabularies
$FASTBPE_PATH getvocab $BASE_PATH/bpe/train.$SRC_LANG > $BASE_PATH/bpe/vocab.$VOCAB_SIZE.$SRC_LANG
$FASTBPE_PATH getvocab $BASE_PATH/bpe/train.$TRG_LANG > $BASE_PATH/bpe/vocab.$VOCAB_SIZE.$TRG_LANG
