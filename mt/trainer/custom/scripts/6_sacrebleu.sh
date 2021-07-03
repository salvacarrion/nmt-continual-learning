#!/bin/bash

# Define constants
EVAL_PATH=$1
OUTPUT_PATH=$2

# Show constants
echo "Evaluating model... ****************"
echo "- Evaluate path: "$EVAL_PATH
echo "- Output path: "$OUTPUT_PATH

# Get metrics
echo "Computing sacrebleu metrics..."
echo "=> Start time (sacrebleu, bleu): $(date)"
cat $EVAL_PATH/hyp.txt | sacrebleu $EVAL_PATH/ref.txt --metrics bleu > $OUTPUT_PATH/metrics_bleu.txt
echo "=> End time (sacrebleu, bleu): $(date)"

echo "=> Start time (sacrebleu, chrf): $(date)"
cat $EVAL_PATH/hyp.txt | sacrebleu $EVAL_PATH/ref.txt --metrics chrf >> $OUTPUT_PATH/metrics_chrf.txt
echo "=> End time (sacrebleu, chrf): $(date)"

#echo "=> Start time (sacrebleu, ter): $(date)"
#cat $EVAL_PATH/hyp.txt | sacrebleu $EVAL_PATH/ref.txt --metrics ter >> $OUTPUT_PATH/metrics_ter.txt
#echo "=> End time (sacrebleu, ter): $(date)"