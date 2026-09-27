# v9 (GPU lane): language-transfer gate for the cross-encoder. Holdout band pairs (work_t xenc2F_v7/train.parquet, locked holdout S1 only, 200k
# sample) with their original texts and with French-ized texts (work_fr parquet), both scored by e5-base (work_t xenc2/model): average precision.
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
until [ -f $SM/work_fr/parquet/train/labels.parquet ]; do sleep 60; done
export SM BER_WORK=$SM/work_fr
ln -sfn $SM/work_t/sample $SM/work_fr/sample  # holdout_q() reads the S1 sample; ids are unchanged in the French copy
python - <<'PY'
import os
from pathlib import Path
import numpy as np, polars as pl
from ber.split import holdout_q
from ber.stages.xenc import _texts, tag_digits
PB = 10_000_000
SM = Path(os.environ["SM"])
d = pl.read_parquet(SM / "work_t" / "xenc2F_v7" / "train.parquet").join(pl.DataFrame({"q": holdout_q().astype(np.int64)}), on="q", how="semi")
d = d.sample(min(200_000, d.height), seed=0)
s1t, poolt, n2 = _texts("train")
pid = d["pid"].to_numpy()
pidx = np.where(pid < 3 * PB, pid - 2 * PB, n2 + pid - 3 * PB)
fr = d.with_columns(pl.Series("ta", [tag_digits(t) for t in s1t[d["q"].to_numpy()]]), pl.Series("tb", [tag_digits(t) for t in poolt[pidx]]))
for name, x in (("xencH0", d), ("xencH", fr)):
    (SM / "work_fr" / name).mkdir(parents=True, exist_ok=True)
    x.write_parquet(SM / "work_fr" / name / "train.parquet")
print("holdout band pairs", d.height, "positives", int(d["label"].sum()))
for a, b, l in list(zip(d["ta"].head(6), fr["ta"].head(6), d["label"].head(6))):
    print(l, "|", a, "=>", b)
PY
for dir in xencH0 xencH; do echo "== $dir"; python -m ber.stages.xenc score --split train --dir $dir --model-dir $SM/work_t/xenc2/model 2>&1 | grep -E "average precision|scored|Error|Traceback"; done
