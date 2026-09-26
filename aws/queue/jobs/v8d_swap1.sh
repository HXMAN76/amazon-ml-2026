# v8 step d (CPU lane jobs2, waits for model 1's scores): the France-aware cross-encoder 1 score replaces xs for France's pairs only; the s22
# stack is rebuilt on the test split with it (same flags as s22) and predicts s22F1 with s22's own model; US and India must come out unchanged.
# Then label-free checks of the new France probabilities and the t2c rule recipe (plus the per-S1 protect option) on s22F1.
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
X=${X:-1}
while [ ! -f $BER_WORK/xencF$X/train_xs.parquet ]; do sleep 60; done
python src/scripts/xfr_report.py s22 xencF$X
python src/scripts/xs_merge.py xencM$X xenc2F_v7 xencF$X
python -m ber.stages.stack build --split test --base v7 --tag _pF$X --xenc --xcons --xenc-fit-more 300000 --decoy --extra --xenc-dir xencM$X --xenc-dir2 xenc_v7
python -m ber.stages.stack tfidf --split test --tag _pF$X
mkdir -p $BER_WORK/models/s22F$X
cp $BER_WORK/models/s22/{xgb.json,holdout.json,holdout_pred.parquet,oof_tune.parquet} $BER_WORK/models/s22F$X/
python -c "import json,sys; c=json.load(open('$BER_WORK/models/s22/config.json')); print('s22 tag', c.get('tag')); c['tag']='_pF$X'; json.dump(c, open('$BER_WORK/models/s22F$X/config.json','w'))"
python -m ber.stages.stack predict --name s22F$X
python - <<PY
import polars as pl
W = "$BER_WORK"
s1 = pl.read_parquet(f"{W}/parquet/test/source1.parquet", columns=["rid", "ctry"]).select(pl.col("rid").cast(pl.Int64).alias("q"), "ctry")
a = pl.read_parquet(f"{W}/output/s22/pair_p.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
b = pl.read_parquet(f"{W}/output/s22F$X/pair_p.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64)).rename({"p": "p_new"})
m = a.join(b, on=["q", "pid"], how="full", coalesce=True).join(s1, on="q", how="left").with_columns((pl.col("p_new") - pl.col("p")).abs().alias("dp"))
print(m.group_by("ctry").agg(pl.len(), pl.col("dp").max().alias("max_dp"), pl.col("dp").mean().alias("mean_dp"), pl.col("p").is_null().sum().alias("missing_old"), pl.col("p_new").is_null().sum().alias("missing_new")))
PY
python src/scripts/france_empty.py s22 s22F$X
python src/scripts/decoy_by_pzone.py s22F$X
python src/scripts/france_variants.py s22F$X v8F${X}_t2c --rules typeswap:1.01,thr:0.985 --cap
python src/scripts/france_variants.py s22F$X v8F${X}_t2cp --rules typeswap:1.01,thr:0.985,protect:0.9 --cap
python src/scripts/france_empty.py s22 v8F${X}_t2c
