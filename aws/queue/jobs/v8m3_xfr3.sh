# v8 step m3 (GPU lane jobs): France-aware cross-encoder 3, fixing model 1's faults (it also rejected true noise-word swaps, more exact names and
# more "other" pairs): the FULL replay of the team's fit set, fewer sibling swaps (100k) balanced by 50k swaps into train's own noise words as
# true copies, plus initials / spelled legal forms / glued words. Then the report, the France-only swap of s22's stack and the recipe.
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
python -m pytest -q src/tests/test_xenc_fr.py
python -m ber.stages.xenc_fr --base-dir xenc2_v7 --all-dir xenc2F_v7 --dir xencFR3 --set n_replay=2000000,n_swap=100000,n_noise=50000
python -m ber.stages.xenc train --dir xencFR3 --base-model $BER_WORK/xenc2/model --model-dir xencFR3/model --set epochs=1,lr=0.00001,batch=64
mkdir -p $BER_WORK/xencF3
ln -sf $BER_WORK/xencFR3/test.parquet $BER_WORK/xencF3/test.parquet
ln -sf $BER_WORK/xencFR3/train.parquet $BER_WORK/xencF3/train.parquet
python -m ber.stages.xenc score --split test --dir xencF3 --model-dir xencFR3/model --set score_batch=512
python -m ber.stages.xenc score --split train --dir xencF3 --model-dir xencFR3/model --set score_batch=512
python src/scripts/xfr_report.py s22 xencF3
python src/scripts/xfr_slots.py s22 xencF3
python src/scripts/xs_merge.py xencM3n xenc2F_v7 xencF3 --keep-old-noise
python -m ber.stages.stack build --split test --base v7 --tag _pF3n --xenc --xcons --xenc-fit-more 300000 --decoy --extra --xenc-dir xencM3n --xenc-dir2 xenc_v7 --ctry france
python -m ber.stages.stack tfidf --split test --tag _pF3n
python src/scripts/stack_predict_merge.py s22F3n s22 _pF3n
python src/scripts/france_empty.py s22 s22F3n
python src/scripts/france_variants.py s22F3n v8_s22F3n_tpp --rules typeswap:1.01,thrp:0.985,protect:0.9 --cap
