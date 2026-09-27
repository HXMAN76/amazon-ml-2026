# v9 (GPU lane): French-aware e5-base cross-encoder. Fit set = 400k pairs of the e5-base fit set (xenc2_v7/train_fit.parquet, training S1 only,
# holdout excluded by construction) with original text + 600k of the same fit pairs with French-ized text (work_fr parquet); warm start from
# xenc2/model, 1 epoch; then the gate: both holdout sets (work_fr xencH0 original, xencH French-ized) scored with the new model.
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export SM
BER_WORK=$SM/work_fr python - <<'PY'
import os
from pathlib import Path
import numpy as np, polars as pl
from ber.stages.xenc import _texts, tag_digits
PB = 10_000_000
SM = Path(os.environ["SM"])
fit = pl.read_parquet(SM / "work_t" / "xenc2_v7" / "train_fit.parquet").select("q", "pid", "label", "ta", "tb")
rng = np.random.default_rng(7)
idx = rng.permutation(fit.height)
orig = fit[idx[:400_000]]
fr = fit[idx[-600_000:]]
s1t, poolt, n2 = _texts("train")
pid = fr["pid"].to_numpy()
pidx = np.where(pid < 3 * PB, pid - 2 * PB, n2 + pid - 3 * PB)
fr = fr.with_columns(pl.Series("ta", [tag_digits(t) for t in s1t[fr["q"].to_numpy()]]), pl.Series("tb", [tag_digits(t) for t in poolt[pidx]]))
out = SM / "work_t" / "xencFZ"
out.mkdir(parents=True, exist_ok=True)
pl.concat([orig, fr]).sample(fraction=1.0, seed=3).write_parquet(out / "train_fit.parquet")
print("fit pairs", orig.height + fr.height, "positives", int(orig["label"].sum() + fr["label"].sum()), "of which French-ized", fr.height, flush=True)
PY
export BER_WORK=$SM/work_t
python -m ber.stages.xenc train --dir xencFZ --base-model $SM/work_t/xenc2/model --model-dir xencFZ/model --set epochs=1,lr=0.00001,batch=64 2>&1 | grep -vE "Warning|warn" | tail -5
export BER_WORK=$SM/work_fr
for dir in xencH0 xencH; do echo "== $dir with xencFZ"; python -m ber.stages.xenc score --split train --dir $dir --model-dir $SM/work_t/xencFZ/model 2>&1 | grep -E "average precision|scored|Error|Traceback"; done
