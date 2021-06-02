#!/bin/bash

# Define constants
VOCAB_SIZE=$1
SRC_LANG=$2
TRG_LANG=$3
MODEL_BASEPATH=$4
TEST_DATAPATH=$5
OUTPUT_PATH=$6
FASTBPE_PATH=$7

# Show constants
echo "Preprocessing files for Fairseq... ****************"
echo "- Source language: "$SRC_LANG
echo "- Target language: "$TRG_LANG
echo "- Model path: "$MODEL_BASEPATH
echo "- Testset path: "$TEST_DATAPATH
echo "- Output path: "$OUTPUT_PATH

# Create folder
mkdir -p "${OUTPUT_PATH}"
mkdir -p "${OUTPUT_PATH}/bpe"

# Apply BPE
$FASTBPE_PATH applybpe $OUTPUT_PATH/bpe/test.tok.bpe.$VOCAB_SIZE.$SRC_LANG $TEST_DATAPATH/clean/test.tok.clean.$SRC_LANG $MODEL_BASEPATH/bpe/bpecodes
$FASTBPE_PATH applybpe $OUTPUT_PATH/bpe/test.tok.bpe.$VOCAB_SIZE.$TRG_LANG $TEST_DATAPATH/clean/test.tok.clean.$TRG_LANG $MODEL_BASEPATH/bpe/bpecodes

# Preprocess files
fairseq-preprocess \
    --source-lang $SRC_LANG --target-lang $TRG_LANG \
    --testpref $OUTPUT_PATH/bpe/test.tok.bpe.$VOCAB_SIZE \
    --srcdict	$MODEL_BASEPATH/data-bin/dict.${SRC_LANG}.txt \
    --tgtdict	$MODEL_BASEPATH/data-bin/dict.${TRG_LANG}.txt \
    --destdir $OUTPUT_PATH/data-bin \
    --workers $(nproc) \

