# v8 (CPU lane jobs2): US/India threshold. s28's holdout macro F0.5 by threshold; then v8u_s28_AR with US and India at 0.80 (France unchanged) for a
# portal reading of the test's higher distractor density (41% unowned pool records against 26% in train); validation --check-ids and upload.
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
python - <<PY
import json, numpy as np, polars as pl
from ber import config, decision
from ber.split import holdout_q
P = config.paths(); m = P["work"] / "models" / "s28"
hq = pl.DataFrame({"q": holdout_q().astype(np.int64)})
lab = pl.read_parquet(P["parquet"] / "train" / "labels.parquet").group_by("s1_rid").len().rename({"s1_rid": "q", "len": "n_true"}).with_columns(pl.col("q").cast(pl.Int64))
nt = hq.join(lab, on="q", how="left").with_columns(pl.col("n_true").fill_null(0))
ex = decision.assign_exclusive(pl.read_parquet(m / "holdout_pred.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64)).join(hq, on="q", how="semi"))
for t in (0.66, 0.70, 0.72, 0.75, 0.80, 0.85, 0.90):
    print(f"threshold {t:.2f}: holdout macro F0.5 {decision.macro_f05(ex.filter(pl.col('p') >= t), nt):.6f}")
PY
mkdir -p $BER_WORK/models/v8u_s28_AR && cp $BER_WORK/models/s28/config.json $BER_WORK/models/v8u_s28_AR/config.json
python src/scripts/reemit.py v8u_s28_AR v8u_s28_AR_t80 --thr us=0.80,india=0.80 2>&1 | grep -E "kept|wrote|PASS|FAIL|Traceback"
o=$BER_WORK/output/v8u_s28_AR_t80
python $BER_WORK/official/validate_submission.py --matching $o/matching_results.tsv --candidate $o/candidate_pairs.tsv --test-dir $BER_DATA/test --check-ids | tail -3 | grep -q "^PASS" \
  && { echo "v8u_s28_AR_t80: PASS with --check-ids"; aws s3 cp $o/ s3://$B/ber/v8/runs/v8u_s28_AR_t80/output/ --recursive --exclude "*" --include "*.tsv" --only-show-errors; } || echo "v8u_s28_AR_t80: VALIDATOR FAILED"
