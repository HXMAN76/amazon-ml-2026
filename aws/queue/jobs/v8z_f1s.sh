# v8 fast swap (runs on the GPU lane jobs after model 2, the GPU is idle then): France's cross-encoder scores := xencF1 only on one-word swaps into a non-noise word (every other pair keeps the old score); the s22 stack is rebuilt for
# France's S1 only, scored with s22's own model and merged into s22's probabilities (US and India unchanged by construction) -> s22F1s;
# then label-free checks and the France recipes on it.
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t

python src/scripts/xs_merge.py xencM1s xenc2F_v7 xencF1 --keep-old-noise --only-swaps
python -m ber.stages.stack build --split test --base v7 --tag _pF1s --xenc --xcons --xenc-fit-more 300000 --decoy --extra --xenc-dir xencM1s --xenc-dir2 xenc_v7 --ctry france
python -m ber.stages.stack tfidf --split test --tag _pF1s
python src/scripts/stack_predict_merge.py s22F1s s22 _pF1s
python src/scripts/france_empty.py s22 s22F1s
python src/scripts/decoy_by_pzone.py s22F1s
python src/scripts/france_variants.py s22F1s v8F1s_ts --rules typeswap:1.01 --cap
python src/scripts/france_variants.py s22F1s v8F1s_tpp --rules typeswap:1.01,thrp:0.985,protect:0.9 --cap
