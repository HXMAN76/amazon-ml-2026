# v9 FAST (CPU lane, GPU shared): France pairs scored by the French-aware cross-encoder xencFZ; thresholds from the labelled French-ized holdout
# (drop: xsF below the largest t where >= 85% of holdout pairs with p1 >= 0.72 are false; add: xsF above the smallest t where >= 97% of holdout
# pairs with p1 < 0.72 are true, never siblings / legal-form conflicts); applied on top of the FIN recipe on s28; validation, upload.
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export SM BER_WORK=$SM/work_t
python - <<'PY'
import os
from pathlib import Path
import polars as pl
W = Path(os.environ["SM"]) / "work_t"
fr = pl.read_parquet(W / "parquet" / "test" / "source1.parquet", columns=["rid", "ctry"]).filter(pl.col("ctry") == "france").select(pl.col("rid").cast(pl.Int64).alias("q"))
d = pl.read_parquet(W / "xenc2F_v7" / "test.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64)).join(fr, on="q", how="semi")
(W / "xencFZfr").mkdir(parents=True, exist_ok=True)
d.write_parquet(W / "xencFZfr" / "test.parquet")
print("France pairs to score", d.height, flush=True)
PY
python -m ber.stages.xenc score --split test --dir xencFZfr --model-dir xencFZ/model --set score_batch=512 2>&1 | grep -E "scored|Error|Traceback"
python - <<'PY'
import os, sys
from pathlib import Path
import numpy as np, polars as pl
sys.path.insert(0, "src/scripts")
from ber import decision
from band_kinds import kinds
from france_variants import legal_set
PB = 10_000_000
SM = Path(os.environ["SM"]); W = SM / "work_t"; F = SM / "work_fr"
h = pl.read_parquet(F / "xencH" / "train.parquet").select("q", "pid", "p", "label").join(pl.read_parquet(F / "xencH" / "train_xs.parquet"), on=["q", "pid"])
hi, lo = h.filter(pl.col("p") >= 0.72), h.filter(pl.col("p") < 0.72)
print("French-ized holdout calibration (xencFZ):")
for t in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3):
    x = hi.filter(pl.col("xs") < t); print(f"  p1>=0.72 & xsF<{t}: {x.height} pairs, false share {1 - x['label'].mean() if x.height else float('nan'):.3f}")
for t in (0.9, 0.95, 0.98, 0.99, 0.995, 0.999):
    x = lo.filter(pl.col("xs") > t); print(f"  p1<0.72 & xsF>{t}: {x.height} pairs, true share {x['label'].mean() if x.height else float('nan'):.3f}")
td = max([t for t in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3) if hi.filter(pl.col("xs") < t).height >= 50 and 1 - hi.filter(pl.col("xs") < t)["label"].mean() >= 0.85] or [0.0])
ta = min([t for t in (0.9, 0.95, 0.98, 0.99, 0.995, 0.999) if lo.filter(pl.col("xs") > t).height >= 50 and lo.filter(pl.col("xs") > t)["label"].mean() >= 0.97] or [1.01])
print(f"chosen: drop xsF < {td}, add xsF > {ta}", flush=True)
xs = pl.read_parquet(W / "xencFZfr" / "test_xs.parquet").select(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64), pl.col("xs").alias("xsF"))
pq = W / "parquet" / "test"
s1 = pl.read_parquet(pq / "source1.parquet", columns=["rid", "entity_id", "core1", "name1", "legal"]).select(pl.col("rid").cast(pl.Int64).alias("q"), pl.col("entity_id").alias("e1"), pl.col("core1").alias("a_core"), pl.col("name1").alias("a_name"), legal_set(pl.col("name1"), pl.col("legal")).alias("la"))
pool = pl.concat([pl.read_parquet(pq / f"source{s}.parquet", columns=["rid", "entity_id", "core1", "name1", "legal"]).select((pl.col("rid").cast(pl.Int64) + s * PB).alias("pid"), pl.col("entity_id").alias("eb"), pl.col("core1").alias("b_core"), pl.col("name1").alias("b_name"), legal_set(pl.col("name1"), pl.col("legal")).alias("lb")) for s in (2, 3)])
m = pl.read_csv(W / "output" / "v8u_s28_FIN" / "matching_results.tsv", separator="\t", infer_schema=False).fill_null("")
kept = m.filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").rename({"source1_entity_id": "e1", "matched_entity_ids": "eb"}).join(s1.select("q", "e1"), on="e1").join(pool.select("pid", "eb"), on="eb").select("q", "pid")
d = xs.join(s1, on="q").join(pool, on="pid")
d = kinds(d).with_columns(((pl.col("la").list.len() > 0) & (pl.col("lb").list.len() > 0) & (pl.col("la").list.set_intersection(pl.col("lb")).list.len() == 0)).alias("legal_disj"))
k = d.join(kept.with_columns(pl.lit(True).alias("kept")), on=["q", "pid"], how="left").with_columns(pl.col("kept").fill_null(False))
drop = k.filter(pl.col("kept") & (pl.col("xsF") < td))
owned = kept.select("pid")
add = k.filter(~pl.col("kept") & (pl.col("xsF") > ta) & ~(pl.col("kind") == "one_swap") & ~pl.col("legal_disj")).join(owned, on="pid", how="anti")
add = add.sort("xsF", descending=True).unique("pid", keep="first")
drop.select("q", "pid").write_parquet(W / "v8x" / "xf_drop.parquet"); add.select("q", "pid").write_parquet(W / "v8x" / "xf_add.parquet")
with pl.Config(tbl_rows=20, fmt_str_lengths=40, tbl_width_chars=200):
    print(f"France FIN-kept pairs with xsF < {td}: {drop.height}; by kind {sorted(drop.group_by('kind').len().rows(), key=lambda r: -r[1])[:6]}")
    print(drop.sample(min(15, drop.height), seed=0).select("xsF", "kind", "a_name", "b_name"))
    print(f"France pairs not in FIN with xsF > {ta} (no siblings / legal conflicts, record unowned): {add.height}; by kind {sorted(add.group_by('kind').len().rows(), key=lambda r: -r[1])[:6]}")
    print(add.sample(min(15, add.height), seed=0).select("xsF", "kind", "a_name", "b_name"))
PY
python src/scripts/france_variants.py s28 v9_xF_FIN --rules "typeswap:1.01,typeswap:1.01:0.6:30:300,thrpn:0.9999,legalx:1.01,droplist:fb_ns_ref,droplist:fb_nsnear_ref,droplist:xf_drop,protect:0.9,restore:noise_swap+noise_extra+initials+spelled_legal+glued:0.05,addlist:fb_coined_hi,addlist:xf_add" --cap 2>&1 | grep -E "^rule droplist:xf|^addlist|total dropped|wrote|Traceback" | cut -c1-200
n=v9_xF_FIN; o=$BER_WORK/output/$n
python $BER_WORK/official/validate_submission.py --matching $o/matching_results.tsv --candidate $o/candidate_pairs.tsv --test-dir $BER_DATA/test --check-ids | tail -3 | grep -q "^PASS" \
  && { echo "$n: PASS with --check-ids"; aws s3 cp $o/ s3://$B/ber/v8/runs/$n/output/ --recursive --exclude "*" --include "*.tsv" --only-show-errors; } || echo "$n: VALIDATOR FAILED"
