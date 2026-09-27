# v9 (CPU lane jobs2, shares the GPU with the training in lane jobs): French-ized holdout for the full stack. Holdout pairs of each cross-encoder
# pair list (xenc2F_v7: xs sym seeds + xs3, xenc_v7: xs2, xenc3Q_v7: xs4) rewritten with French-ized text (work_fr), re-scored by the same models;
# their scores replace xs / xs_asym / xs_seed_gap / xs2 / xs3 / xs4 in s28's holdout rows (stack_eq/train chunks; other features unchanged); s28
# predicts; macro F0.5 against the original holdout 0.990770.
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export SM
for d in xenc2sym xenc2sym2 xenc; do aws s3 sync s3://$B/ber/team_work/$d/model $SM/work_t/$d/model --only-show-errors; done
ln -sfn $SM/work_t/sample $SM/work_fr/sample
BER_WORK=$SM/work_fr python - <<'PY'
import os
from pathlib import Path
import numpy as np, polars as pl
from ber.split import holdout_q
from ber.stages.xenc import _texts, tag_digits
PB = 10_000_000
SM = Path(os.environ["SM"])
hq = holdout_q().astype(np.int64).tolist()
cols = ["q", "pid", "p", "label", "xs3", "xs2", "xs4"]
h = pl.concat([pl.read_parquet(f, columns=cols).with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64)).filter(pl.col("q").is_in(hq))
               for f in sorted((SM / "work_t" / "stack_eq" / "train").glob("chunk_*.parquet"))])
s1t, poolt, n2 = _texts("train")
for feat, dsts in (("xs3", ["hF2F", "hF2Fs1", "hF2Fs2"]), ("xs2", ["hFs"]), ("xs4", ["hFQ"])):
    d = h.filter(pl.col(feat).is_not_null()).select("q", "pid", "p", "label")
    pid = d["pid"].to_numpy()
    pidx = np.where(pid < 3 * PB, pid - 2 * PB, n2 + pid - 3 * PB)
    d = d.with_columns(pl.Series("ta", [tag_digits(t) for t in s1t[d["q"].to_numpy()]]), pl.Series("tb", [tag_digits(t) for t in poolt[pidx]]))
    for sub in dsts:
        (SM / "work_fr" / sub).mkdir(parents=True, exist_ok=True)
        d.write_parquet(SM / "work_fr" / sub / "train.parquet")
    print(feat, "holdout pairs to re-score", d.height, flush=True)
PY
export BER_WORK=$SM/work_fr
M=$SM/work_t
python -m ber.stages.xenc score --split train --dir hF2Fs1 --model-dir $M/xenc2sym/model --set score_batch=512,symmetric=1 2>&1 | grep -E "scored|Error|Traceback"
python -m ber.stages.xenc score --split train --dir hF2Fs2 --model-dir $M/xenc2sym2/model --set score_batch=512,symmetric=1 2>&1 | grep -E "scored|Error|Traceback"
python -m ber.stages.xenc score --split train --dir hF2F --model-dir $M/xenc2/model --set score_batch=512 2>&1 | grep -E "scored|Error|Traceback"
python -m ber.stages.xenc score --split train --dir hFs --model-dir $M/xenc/model 2>&1 | grep -E "scored|Error|Traceback"
python -m ber.stages.xenc score --split train --dir hFQ --model-dir $M/xenc3Q/model --set score_batch=256,max_len=192 2>&1 | grep -E "scored|Error|Traceback"
python - <<'PY'
import json, os
from pathlib import Path
import numpy as np, polars as pl, xgboost as xgb
from ber import decision
from ber.split import holdout_q
SM = Path(os.environ["SM"]); F = SM / "work_fr"; T = SM / "work_t"
lg = lambda x: np.log(np.clip(x, 1e-7, 1 - 1e-7) / (1 - np.clip(x, 1e-7, 1 - 1e-7)))
a = pl.read_parquet(F / "hF2Fs1" / "train_xs.parquet"); b = pl.read_parquet(F / "hF2Fs2" / "train_xs.parquet").select("q", "pid", pl.col("xs").alias("xb"), pl.col("xs_asym").alias("ab"))
m = a.join(b, on=["q", "pid"])
la, lb = lg(m["xs"].to_numpy()), lg(m["xb"].to_numpy())
sym = m.select(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64)).with_columns(pl.Series("xs_f", (1 / (1 + np.exp(-(la + lb) / 2))).astype(np.float32)),
      ((m["xs_asym"] + m["ab"]) / 2).cast(pl.Float32).alias("xs_asym_f"), pl.Series("xs_seed_gap_f", np.abs(la - lb).astype(np.float32)))
rd = lambda d, c: pl.read_parquet(F / d / "train_xs.parquet").select(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64), pl.col("xs").alias(c))
fr = sym.join(rd("hF2F", "xs3_f"), on=["q", "pid"], how="full", coalesce=True).join(rd("hFs", "xs2_f"), on=["q", "pid"], how="full", coalesce=True).join(rd("hFQ", "xs4_f"), on=["q", "pid"], how="full", coalesce=True)
hq = holdout_q().astype(np.int64)
rows = [pl.read_parquet(f).with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64)).filter(pl.col("q").is_in(hq.tolist())) for f in sorted((T / "stack_eq" / "train").glob("chunk_*.parquet"))]
h = pl.concat(rows).join(fr, on=["q", "pid"], how="left")
swap = {"xs": "xs_f", "xs_asym": "xs_asym_f", "xs_seed_gap": "xs_seed_gap_f", "xs2": "xs2_f", "xs3": "xs3_f", "xs4": "xs4_f"}
hf = h.with_columns([pl.when(pl.col(o).is_not_null()).then(pl.col(n)).otherwise(pl.col(o)).alias(o) for o, n in swap.items() if o in h.columns])
lab = pl.read_parquet(SM / "work_t" / "parquet" / "train" / "labels.parquet").group_by("s1_rid").len().rename({"s1_rid": "q", "len": "n_true"}).with_columns(pl.col("q").cast(pl.Int64))
nt = pl.DataFrame({"q": hq}).join(lab, on="q", how="left").with_columns(pl.col("n_true").fill_null(0))
for name in ("s28",):
    cfg = json.loads((T / "models" / name / "config.json").read_text()); thr = json.loads((T / "models" / name / "holdout.json").read_text())["stack_threshold"]
    bst = xgb.Booster(model_file=str(T / "models" / name / "xgb.json"))
    for tag, d in (("original", h), ("French-ized cross-encoder scores", hf)):
        p = bst.predict(xgb.DMatrix(d.select(cfg["features"]).to_numpy().astype(np.float32), feature_names=cfg["features"]))
        pr = decision.assign_exclusive(d.select("q", "pid", "label").with_columns(pl.Series("p", p))).filter(pl.col("p") >= thr)
        print(f"{name} holdout {tag}: macro F0.5 {decision.macro_f05(pr, nt):.6f} (kept {pr.height}, wrong {pr.filter(pl.col('label') == 0).height})", flush=True)
fr.write_parquet(F / "holdout_fr_xs.parquet")
PY
