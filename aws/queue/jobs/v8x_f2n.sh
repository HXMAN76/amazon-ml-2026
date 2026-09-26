# v8 fast swap (CPU lane jobs2): France's cross-encoder scores := xencF2 except noise-word swaps (old score kept); the s22 stack is rebuilt for
# France's S1 only, scored with s22's own model and merged into s22's probabilities (US and India unchanged by construction) -> s22F2n;
# then label-free checks and the France recipes on it.
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
while [ ! -f $BER_WORK/xencF2/train_xs.parquet ]; do sleep 60; done
python src/scripts/xs_merge.py xencM2n xenc2F_v7 xencF2 --keep-old-noise
python -m ber.stages.stack build --split test --base v7 --tag _pF2n --xenc --xcons --xenc-fit-more 300000 --decoy --extra --xenc-dir xencM2n --xenc-dir2 xenc_v7 --ctry france
python -m ber.stages.stack tfidf --split test --tag _pF2n
python src/scripts/stack_predict_merge.py s22F2n s22 _pF2n
python src/scripts/france_empty.py s22 s22F2n
python src/scripts/decoy_by_pzone.py s22F2n
python src/scripts/france_variants.py s22F2n v8F2n_ts --rules typeswap:1.01 --cap
python src/scripts/france_variants.py s22F2n v8F2n_t2c --rules typeswap:1.01,thr:0.985 --cap
python src/scripts/france_variants.py s22F2n v8F2n_t2cp --rules typeswap:1.01,thr:0.985,protect:0.9 --cap
