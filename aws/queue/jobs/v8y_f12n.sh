# v8 fast swap (CPU lane jobs2): France's cross-encoder scores := mean logit of xencF1 and xencF2 (xencF12) except noise-word swaps (old score kept); the s22 stack is rebuilt for
# France's S1 only, scored with s22's own model and merged into s22's probabilities (US and India unchanged by construction) -> s22F12n;
# then label-free checks and the France recipes on it.
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
while [ ! -f $BER_WORK/xencF2/train_xs.parquet ] || [ ! -f $BER_WORK/xencF1/train_xs.parquet ]; do sleep 60; done
mkdir -p $BER_WORK/xencF12
python - <<PY
import numpy as np, polars as pl
W = "$BER_WORK"
for split in ("test", "train"):
    a = pl.read_parquet(f"{W}/xencF1/{split}_xs.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    b = pl.read_parquet(f"{W}/xencF2/{split}_xs.parquet").select(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64), pl.col("xs").alias("xs2"))
    m = a.join(b, on=["q", "pid"], how="inner")
    lg = lambda x: np.log(np.clip(x, 1e-6, 1 - 1e-6) / (1 - np.clip(x, 1e-6, 1 - 1e-6)))
    z = (lg(m["xs"].to_numpy()) + lg(m["xs2"].to_numpy())) / 2
    m.select("q", "pid").with_columns(pl.Series("xs", (1 / (1 + np.exp(-z))).astype(np.float32))).write_parquet(f"{W}/xencF12/{split}_xs.parquet")
    print(split, a.height, b.height, m.height)
PY
python src/scripts/xs_merge.py xencM12n xenc2F_v7 xencF12 --keep-old-noise
python -m ber.stages.stack build --split test --base v7 --tag _pF12n --xenc --xcons --xenc-fit-more 300000 --decoy --extra --xenc-dir xencM12n --xenc-dir2 xenc_v7 --ctry france
python -m ber.stages.stack tfidf --split test --tag _pF12n
python src/scripts/stack_predict_merge.py s22F12n s22 _pF12n
python src/scripts/france_empty.py s22 s22F12n
python src/scripts/decoy_by_pzone.py s22F12n
python src/scripts/france_variants.py s22F12n v8F12n_ts --rules typeswap:1.01 --cap
python src/scripts/france_variants.py s22F12n v8F12n_t2c --rules typeswap:1.01,thr:0.985 --cap
python src/scripts/france_variants.py s22F12n v8F12n_t2cp --rules typeswap:1.01,thr:0.985,protect:0.9 --cap
