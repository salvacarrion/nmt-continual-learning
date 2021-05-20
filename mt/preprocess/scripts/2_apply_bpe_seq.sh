#!/bin/sh

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

# Apply BPE ****
# Source
$FASTBPE_PATH applybpe $SAVE_PATH/train.$SRC_LANG $BASE_PATH/clean/train.$SRC_LANG $SAVE_PATH/codes.$SRC_LANG
$FASTBPE_PATH applybpe $SAVE_PATH/val.$SRC_LANG $BASE_PATH/clean/val.$SRC_LANG $SAVE_PATH/codes.$SRC_LANG
$FASTBPE_PATH applybpe $SAVE_PATH/test.$SRC_LANG $BASE_PATH/clean/test.$SRC_LANG $SAVE_PATH/codes.$SRC_LANG

# Target
$FASTBPE_PATH applybpe $SAVE_PATH/train.$TRG_LANG $BASE_PATH/clean/train.$TRG_LANG $SAVE_PATH/codes.$TRG_LANG
$FASTBPE_PATH applybpe $SAVE_PATH/val.$TRG_LANG $BASE_PATH/clean/val.$TRG_LANG $SAVE_PATH/codes.$TRG_LANG
$FASTBPE_PATH applybpe $SAVE_PATH/test.$TRG_LANG $BASE_PATH/clean/test.$TRG_LANG $SAVE_PATH/codes.$TRG_LANG

# Save vocabularies
#$FASTBPE_PATH getvocab $SAVE_PATH/train.$SRC_LANG > $SAVE_PATH/vocab.$SRC_LANG
#$FASTBPE_PATH getvocab $SAVE_PATH/train.$TRG_LANG > $SAVE_PATH/vocab.$TRG_LANG
