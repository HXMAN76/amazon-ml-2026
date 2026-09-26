# v8 step m4 (GPU lane jobs): cross-encoder 4 = the recipe of model 3 with other seeds (synthetic sampling and training), for averaging; fixing model 1's faults (it also rejected true noise-word swaps, more exact names and
# more "other" pairs): the FULL replay of the team's fit set, fewer sibling swaps (100k) balanced by 50k swaps into train's own noise words as
# true copies, plus initials / spelled legal forms / glued words. Then the report, the France-only swap of s22's stack and the recipe.
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
python -m pytest -q src/tests/test_xenc_fr.py
python -m ber.stages.xenc_fr --base-dir xenc2_v7 --all-dir xenc2F_v7 --dir xencFR4 --set n_replay=2000000,n_swap=100000,n_noise=50000,seed=11
python -m ber.stages.xenc train --dir xencFR4 --base-model $BER_WORK/xenc2/model --model-dir xencFR4/model --set epochs=1,lr=0.00001,batch=64,train_seed=1
mkdir -p $BER_WORK/xencF4
ln -sf $BER_WORK/xencFR4/test.parquet $BER_WORK/xencF4/test.parquet
ln -sf $BER_WORK/xencFR4/train.parquet $BER_WORK/xencF4/train.parquet
python -m ber.stages.xenc score --split test --dir xencF4 --model-dir xencFR4/model --set score_batch=512
python -m ber.stages.xenc score --split train --dir xencF4 --model-dir xencFR4/model --set score_batch=512
python src/scripts/xfr_report.py s22 xencF4
python src/scripts/xfr_slots.py s22 xencF4
python src/scripts/xs_merge.py xencM4n xenc2F_v7 xencF4 --keep-old-noise
python -m ber.stages.stack build --split test --base v7 --tag _pF4n --xenc --xcons --xenc-fit-more 300000 --decoy --extra --xenc-dir xencM4n --xenc-dir2 xenc_v7 --ctry france
python -m ber.stages.stack tfidf --split test --tag _pF4n
python src/scripts/stack_predict_merge.py s22F4n s22 _pF4n
python src/scripts/france_empty.py s22 s22F4n
python src/scripts/france_variants.py s22F4n v8_s22F4n_tpp --rules typeswap:1.01,thrp:0.985,protect:0.9 --cap
