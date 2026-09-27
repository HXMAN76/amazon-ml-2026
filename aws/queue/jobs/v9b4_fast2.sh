# v9 FAST 2 (CPU lane): v9_xF_FIN without its legal-form artefact. Drops only FIN-kept France pairs with xsF < 0.05 whose names differ by real
# words (one_swap / words_added / other kinds where the differing words are not all legal forms); adds unchanged (xsF > 0.95, no siblings or
# legal-form conflicts). Validation --check-ids, upload.
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export SM BER_WORK=$SM/work_t
python - <<'PY'
import os, sys
from pathlib import Path
import polars as pl
sys.path.insert(0, "src/scripts")
from band_kinds import kinds
PB = 10_000_000
W = Path(os.environ["SM"]) / "work_t"; pq = W / "parquet" / "test"
LEG = ["sarl", "sas", "sasu", "eurl", "sa", "sci", "snc", "ei", "eirl", "scop", "cie", "compagnie", "s", "a", "r", "l", "u", "e", "i", "c", "n"]
dr = pl.read_parquet(W / "v8x" / "xf_drop.parquet")
s1 = pl.read_parquet(pq / "source1.parquet", columns=["rid", "core1", "name1"]).select(pl.col("rid").cast(pl.Int64).alias("q"), pl.col("core1").alias("a_core"), pl.col("name1").alias("a_name"))
pool = pl.concat([pl.read_parquet(pq / f"source{s}.parquet", columns=["rid", "core1", "name1"]).select((pl.col("rid").cast(pl.Int64) + s * PB).alias("pid"), pl.col("core1").alias("b_core"), pl.col("name1").alias("b_name")) for s in (2, 3)])
d = kinds(dr.join(s1, on="q").join(pool, on="pid"))
ta, tb = pl.col("a_name").str.split(" ").list.unique(), pl.col("b_name").str.split(" ").list.unique()
diff = ta.list.set_difference(tb).list.concat(tb.list.set_difference(ta))
d = d.with_columns(diff.alias("diff"))
keep_drop = d.filter((pl.col("kind") != "reordered") & ~pl.col("diff").list.eval(pl.element().is_in(LEG)).list.all())
keep_drop.select("q", "pid").write_parquet(W / "v8x" / "xf_drop2.parquet")
print("drops kept:", keep_drop.height, "of", d.height, sorted(keep_drop.group_by("kind").len().rows(), key=lambda r: -r[1])[:6], flush=True)
with pl.Config(tbl_rows=12, fmt_str_lengths=40, tbl_width_chars=200):
    print(keep_drop.sample(min(12, keep_drop.height), seed=1).select("kind", "a_name", "b_name"))
PY
python src/scripts/france_variants.py s28 v9_xF2_FIN --rules "typeswap:1.01,typeswap:1.01:0.6:30:300,thrpn:0.9999,legalx:1.01,droplist:fb_ns_ref,droplist:fb_nsnear_ref,droplist:xf_drop2,protect:0.9,restore:noise_swap+noise_extra+initials+spelled_legal+glued:0.05,addlist:fb_coined_hi,addlist:xf_add" --cap 2>&1 | grep -E "^rule droplist:xf|^addlist|total dropped|wrote|Traceback" | cut -c1-200
n=v9_xF2_FIN; o=$BER_WORK/output/$n
python $BER_WORK/official/validate_submission.py --matching $o/matching_results.tsv --candidate $o/candidate_pairs.tsv --test-dir $BER_DATA/test --check-ids | tail -3 | grep -q "^PASS" \
  && { echo "$n: PASS with --check-ids"; aws s3 cp $o/ s3://$B/ber/v8/runs/$n/output/ --recursive --exclude "*" --include "*.tsv" --only-show-errors; } || echo "$n: VALIDATOR FAILED"
