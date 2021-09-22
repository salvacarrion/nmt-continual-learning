# 0.0 Create new folder with (H_H, B_B, M_M) from another

# 0.1 Create new datasets (training data only; ignore rest)
cp -R health_fairseq_vhealth_es-en/ health_fairseq_vbiological_es-en/
cp -R health_fairseq_vhealth_es-en/ health_fairseq_vmerged_es-en/
cp -R biological_fairseq_vbiological_es-en/ biological_fairseq_vhealth_es-en/
cp -R biological_fairseq_vbiological_es-en/ biological_fairseq_vmerged_es-en/
cp -R merged_fairseq_vmerged_es-en/ merged_fairseq_vhealth_es-en/
cp -R merged_fairseq_vmerged_es-en/ merged_fairseq_vbiological_es-en/

# 1 - Delete folders: eval, logs, checkpoints,...

# 2 - Learn&Apply BPE for: H_H/B_B/M_M

# 3 - Copy tok
cp -R health_fairseq_vhealth_es-en/tok/ biological_fairseq_vhealth_es-en/
cp -R health_fairseq_vhealth_es-en/tok/ merged_fairseq_vhealth_es-en/
cp -R biological_fairseq_vbiological_es-en/tok/ health_fairseq_vbiological_es-en/
cp -R biological_fairseq_vbiological_es-en/tok/ merged_fairseq_vbiological_es-en/
cp -R merged_fairseq_vmerged_es-en/tok/ health_fairseq_vmerged_es-en/
cp -R merged_fairseq_vmerged_es-en/tok/ biological_fairseq_vmerged_es-en/

# 3 - Apply BPE for: H_../B_../M_..  (do not save vocabs!)

# 4 - View sizes
VOCAB_SIZE=128
ll health_fairseq_vhealth_es-en/tok/bpe.$VOCAB_SIZE/
ll biological_fairseq_vbiological_es-en/tok/bpe.$VOCAB_SIZE/
ll merged_fairseq_vmerged_es-en/tok/bpe.$VOCAB_SIZE/

# 5 - Compare
VOCAB_SIZE=128
ll health_fairseq_vbiological_es-en/tok/bpe.$VOCAB_SIZE/
ll health_fairseq_vmerged_es-en/tok/bpe.$VOCAB_SIZE/
ll biological_fairseq_vhealth_es-en/tok/bpe.$VOCAB_SIZE/
ll biological_fairseq_vmerged_es-en/tok/bpe.$VOCAB_SIZE/
ll merged_fairseq_vhealth_es-en/tok/bpe.$VOCAB_SIZE/
ll merged_fairseq_vbiological_es-en/tok/bpe.$VOCAB_SIZE/

# 6 - Preprocess

# 7 - Train!

# Sequential
#-----

cp -R biological_fairseq_vhealth_es-en/ health_biological_fairseq_vhealth_es-en/
cp -R biological_fairseq_vbiological_es-en/ health_biological_fairseq_vbiological_es-en/
cp -R biological_fairseq_vmerged_es-en/ health_biological_fairseq_vmerged_es-en/

rm -R health_biological_fairseq_vhealth_es-en/checkpoints/
rm -R health_biological_fairseq_vbiological_es-en/checkpoints/
rm -R health_biological_fairseq_vmerged_es-en/checkpoints/

cp -R health_fairseq_vhealth_es-en/checkpoints/ health_biological_fairseq_vhealth_es-en/
cp -R health_fairseq_vbiological_es-en/checkpoints/ health_biological_fairseq_vbiological_es-en/
cp -R health_fairseq_vmerged_es-en/checkpoints/ health_biological_fairseq_vmerged_es-en/

mv health_biological_fairseq_vhealth_es-en/checkpoints/checkpoint_best.pt health_biological_fairseq_vhealth_es-en/checkpoints/health_checkpoint_best.pt
mv health_biological_fairseq_vbiological_es-en/checkpoints/checkpoint_best.pt health_biological_fairseq_vbiological_es-en/checkpoints/health_checkpoint_best.pt
mv health_biological_fairseq_vmerged_es-en/checkpoints/checkpoint_best.pt health_biological_fairseq_vmerged_es-en/checkpoints/health_checkpoint_best.pt


# Remove toks


#rm -R health_fairseq_vhealth_es-en/tok/
rm -R health_fairseq_vbiological_es-en/tok/
rm -R health_fairseq_vmerged_es-en/tok/
rm -R biological_fairseq_vhealth_es-en/tok/
#rm -R biological_fairseq_vbiological_es-en/tok/
rm -R biological_fairseq_vmerged_es-en/tok/
rm -R merged_fairseq_vhealth_es-en/tok/
rm -R merged_fairseq_vbiological_es-en/tok/
#rm -R merged_fairseq_vmerged_es-en/tok/