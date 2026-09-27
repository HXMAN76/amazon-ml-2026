# v9 (GPU lane, after v9a6_scoreFZ): s30F = s28L's chunks (s28 features + language-independent features) + xsF (French-aware cross-encoder),
# trained on the provided training labels with s28's settings; paired US/India holdout test; French-ized holdout test (cross-encoder scores
# of the holdout rows re-scored on French-ized text, xsF too) for s28 and s30F; then the FIN France rules on s30F, validation, upload.
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export SM BER_WORK=$SM/work_t
[ -f $SM/work_t/xencFZs/test_xs.parquet ] || { echo "no xsF scores"; exit 1; }
python - <<'PY'
import os
from pathlib import Path
import polars as pl
W = Path(os.environ["SM"]) / "work_t"
for split in ("train", "test"):
    x = pl.read_parquet(W / "xencFZs" / f"{split}_xs.parquet").select(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64), pl.col("xs").alias("xsF"))
    (W / "stack_eqLF" / split).mkdir(parents=True, exist_ok=True)
    for f in sorted((W / "stack_eqL" / split).glob("chunk_*.parquet")):
        d = pl.read_parquet(f).with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64)).join(x, on=["q", "pid"], how="left")
        d.write_parquet(W / "stack_eqLF" / split / f.name, compression="zstd")
    print(split, "xsF coverage", round(d["xsF"].is_not_null().mean(), 4), flush=True)
PY
python -m ber.stages.stack train --name s30F --base v7 --tag _eqLF --set max_depth=9,eta=0.05,rounds=1500,early_stop=50 2>&1 | grep -E "HOLDOUT|OOF|top features|Traceback" | cut -c1-240
python -m ber.stages.stack predict --name s30F 2>&1 | grep -E "kept|Traceback" | tail -2
python src/scripts/paired_models.py s28 s30F 2>&1 | grep -E "delta_B_minus_A|holdout" | head -4
python - <<'PY'
import json, os
from pathlib import Path
import numpy as np, polars as pl, xgboost as xgb
from ber import decision
from ber.split import holdout_q
SM = Path(os.environ["SM"]); F = SM / "work_fr"; T = SM / "work_t"
fr = pl.read_parquet(F / "holdout_fr_xs.parquet").join(pl.read_parquet(F / "hF2FZ" / "train_xs.parquet").select(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64), pl.col("xs").alias("xsF_f")), on=["q", "pid"], how="full", coalesce=True)
hq = holdout_q().astype(np.int64).tolist()
h = pl.concat([pl.read_parquet(f).filter(pl.col("q").is_in(hq)) for f in sorted((T / "stack_eqLF" / "train").glob("chunk_*.parquet"))]).join(fr, on=["q", "pid"], how="left")
swap = {"xs": "xs_f", "xs_asym": "xs_asym_f", "xs_seed_gap": "xs_seed_gap_f", "xs2": "xs2_f", "xs3": "xs3_f", "xs4": "xs4_f", "xsF": "xsF_f"}
hf = h.with_columns([pl.when(pl.col(o).is_not_null()).then(pl.col(n)).otherwise(pl.col(o)).alias(o) for o, n in swap.items()])
lab = pl.read_parquet(T / "parquet" / "train" / "labels.parquet").group_by("s1_rid").len().rename({"s1_rid": "q", "len": "n_true"}).with_columns(pl.col("q").cast(pl.Int64))
nt = pl.DataFrame({"q": hq}).join(lab, on="q", how="left").with_columns(pl.col("n_true").fill_null(0))
for name in ("s28", "s28L", "s30F"):
    cfg = json.loads((T / "models" / name / "config.json").read_text()); thr = json.loads((T / "models" / name / "holdout.json").read_text())["stack_threshold"]
    bst = xgb.Booster(model_file=str(T / "models" / name / "xgb.json"))
    for tag, d in (("original", h), ("French-ized", hf)):
        p = bst.predict(xgb.DMatrix(d.select(cfg["features"]).to_numpy().astype(np.float32), feature_names=cfg["features"]))
        pr = decision.assign_exclusive(d.select("q", "pid", "label").with_columns(pl.Series("p", p))).filter(pl.col("p") >= thr)
        print(f"{name} holdout {tag}: macro F0.5 {decision.macro_f05(pr, nt):.6f} (kept {pr.height}, wrong {pr.filter(pl.col('label') == 0).height})", flush=True)
PY
python src/scripts/france_variants.py s30F v9_s30F_ALL2 --rules "typeswap:1.01,typeswap:1.01:0.6:30:300,thrpn:0.9999,legalx:1.01,protect:0.9,restore:noise_swap+noise_extra+initials+spelled_legal+glued:0.05" --cap 2>&1 | grep -E "^rule|^alias|total dropped|wrote|Traceback" | cut -c1-200
python src/scripts/france_lists.py v9_s30F_ALL2 v8x/F s30F 2>&1 | grep -E "^fb_|Traceback"
python src/scripts/france_variants.py s30F v9_s30F_FIN --rules "typeswap:1.01,typeswap:1.01:0.6:30:300,thrpn:0.9999,legalx:1.01,droplist:F/fb_ns_ref,droplist:F/fb_nsnear_ref,protect:0.9,restore:noise_swap+noise_extra+initials+spelled_legal+glued:0.05,addlist:F/fb_coined_hi" --cap 2>&1 | grep -E "^rule|^alias|^addlist|total dropped|wrote|Traceback" | cut -c1-200
n=v9_s30F_FIN; o=$BER_WORK/output/$n
python $BER_WORK/official/validate_submission.py --matching $o/matching_results.tsv --candidate $o/candidate_pairs.tsv --test-dir $BER_DATA/test --check-ids | tail -3 | grep -q "^PASS" \
  && { echo "$n: PASS with --check-ids"; aws s3 cp $o/ s3://$B/ber/v8/runs/$n/output/ --recursive --exclude "*" --include "*.tsv" --only-show-errors; } || echo "$n: VALIDATOR FAILED"
