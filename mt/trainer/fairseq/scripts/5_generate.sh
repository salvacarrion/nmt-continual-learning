#!/bin/bash

# Define constants
EVAL_PATH=$1
MODEL_PATH=$2
OUTPUT_PATH=$3
SRC_LANG=$4
TRG_LANG=$5
BEAM=$6

# Show constants
echo "Evaluating model... ****************"
echo "- Evaluate path: "$EVAL_PATH
echo "- Model path: "$MODEL_PATH
echo "- Output path: "$OUTPUT_PATH
echo "- Source language: "$SRC_LANG
echo "- Target language: "$TRG_LANG
echo "- Beam width: "$BEAM

echo "=> Start time: $(date)"

# Evaluate model
fairseq-generate \
    $EVAL_PATH \
    --source-lang $SRC_LANG --target-lang $TRG_LANG \
    --path $MODEL_PATH \
    --results-path $OUTPUT_PATH \
    --scoring bleu \
    --beam $BEAM \
    --nbest 1 \
    --max-len-a 1.2 \
    --max-len-b 10 \
    --num-workers $(nproc) \

echo "=> End time: $(date)"


# Clean output
echo "Cleaning output..."
grep ^H $OUTPUT_PATH/generate-test.txt | cut -f3- > $OUTPUT_PATH/hyp.txt
grep ^T $OUTPUT_PATH/generate-test.txt | cut -f2- > $OUTPUT_PATH/ref.txt

# Get metrics
echo "Computing sacrebleu metrics..."
echo "=> Start time (sacrebleu, bleu): $(date)"
cat $OUTPUT_PATH/hyp.txt | sacrebleu $OUTPUT_PATH/ref.txt --metrics bleu > $OUTPUT_PATH/metrics_bleu.txt
echo "=> End time (sacrebleu, bleu): $(date)"

echo "=> Start time (sacrebleu, chrf): $(date)"
cat $OUTPUT_PATH/hyp.txt | sacrebleu $OUTPUT_PATH/ref.txt --metrics chrf >> $OUTPUT_PATH/metrics_chrf.txt
echo "=> End time (sacrebleu, chrf): $(date)"

echo "=> Start time (sacrebleu, ter): $(date)"
cat $OUTPUT_PATH/hyp.txt | sacrebleu $OUTPUT_PATH/ref.txt --metrics ter >> $OUTPUT_PATH/metrics_ter.txt
echo "=> End time (sacrebleu, ter): $(date)"

#fairseq-generate data-bin/scielo_health_es_en/ --source-lang es --target-lang en --path checkpoints/transformer/checkpoint_best.pt --tokenizer moses --remove-bpe --beam 5 --scoring bleu
#fairseq-interactive data-bin/scielo_health_es_en/ --path checkpoints/transformer/checkpoint_best.pt --beam 5 --source-lang es --target-lang en --tokenizer moses --bpe fastbpe --bpe-codes codes
