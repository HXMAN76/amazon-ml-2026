# v8 step m5 (GPU lane jobs, CPU work after model 4): the mean logit of France cross-encoders 3 and 4 as France's score (noise-word swaps keep
# the old one), the s22 stack rebuilt for France only, and the candidate recipes on it.
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
mkdir -p $BER_WORK/xencF34
python - <<PY
import numpy as np, polars as pl
W = "$BER_WORK"
lg = lambda x: np.log(np.clip(x, 1e-6, 1 - 1e-6) / (1 - np.clip(x, 1e-6, 1 - 1e-6)))
for split in ("test", "train"):
    a = pl.read_parquet(f"{W}/xencF3/{split}_xs.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    b = pl.read_parquet(f"{W}/xencF4/{split}_xs.parquet").select(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64), pl.col("xs").alias("xs2"))
    m = a.join(b, on=["q", "pid"], how="inner")
    z = (lg(m["xs"].to_numpy()) + lg(m["xs2"].to_numpy())) / 2
    m.select("q", "pid").with_columns(pl.Series("xs", (1 / (1 + np.exp(-z))).astype(np.float32))).write_parquet(f"{W}/xencF34/{split}_xs.parquet")
    print(split, a.height, b.height, m.height)
PY
python src/scripts/xfr_report.py s22 xencF34
python src/scripts/xs_merge.py xencM34n xenc2F_v7 xencF34 --keep-old-noise
python -m ber.stages.stack build --split test --base v7 --tag _pF34n --xenc --xcons --xenc-fit-more 300000 --decoy --extra --xenc-dir xencM34n --xenc-dir2 xenc_v7 --ctry france
python -m ber.stages.stack tfidf --split test --tag _pF34n
python src/scripts/stack_predict_merge.py s22F34n s22 _pF34n
python src/scripts/france_empty.py s22 s22F34n
R=restore:exact+spelled_legal+initials+glued+noise_swap
python src/scripts/france_variants.py s22F34n v8q_s22F34n_T --rules typeswap:1.01,thrp:0.995,protect:0.9 --cap 2>&1 | grep -E "fires on|^protect|^restore|with matches|PASS|FAIL|Traceback"
python src/scripts/france_variants.py s22F34n v8q_s22F34n_TR --rules typeswap:1.01,thrp:0.995,protect:0.9,$R:0.3 --cap 2>&1 | grep -E "fires on|^protect|^restore|with matches|PASS|FAIL|Traceback"
