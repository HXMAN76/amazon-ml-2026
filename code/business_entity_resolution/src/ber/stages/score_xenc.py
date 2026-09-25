"""score_xenc: apply a fine-tuned cross-encoder (train_xenc) to candidate pairs, band-gated.

Scores only pairs whose baseline pair-model probability falls in the same [lo, hi] band the cross-encoder
was fine-tuned on (config carried in xenc_config.json) — outside that band the pointwise matcher is already
confident, so paying the cross-encoder's inference cost there is wasted compute for no expected gain.

Inputs : WORK/models/<name>/xenc/ (fine-tuned weights), WORK/models/<name>/xenc_config.json
         WORK/features/{split}/part_*.parquet (q, pid, p1 columns come from a separate pair-model score file)
         WORK/parquet/{split}/source{1,2,3}.parquet (raw text)
Outputs: WORK/v2/<name>/xenc_{split}.parquet (q, pid, xs)
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import polars as pl

from ber import config
from ber.stages.train_xenc import _pool_text
from ber.tracking import log_stage


def main(argv: list[str] | None = None) -> None:
    """CLI: score the uncertain band of a split's candidates with the fine-tuned cross-encoder."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="xenc0")
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--pair-p", required=True,
                     help="parquet with columns q, pid, p (e.g. WORK/models/v0base/oof.parquet or output/v0base/pair_p.parquet)")
    ap.add_argument("--batch-size", type=int, default=128)
    a = ap.parse_args(argv)
    P = config.paths()
    t0 = time.time()

    mdl = P["work"] / "models" / a.name
    cfg = json.loads((mdl / "xenc_config.json").read_text())
    scored = pl.read_parquet(a.pair_p)
    band = scored.filter((pl.col("p") >= cfg["lo"]) & (pl.col("p") <= cfg["hi"])).select("q", "pid")
    print(f"scoring {band.height} pairs in band [{cfg['lo']}, {cfg['hi']}] with {cfg['base_model']}", flush=True)

    pq = P["parquet"] / a.split
    cols = ["rid", "core1", "addr"]
    s1 = pl.read_parquet(pq / "source1.parquet", columns=cols).sort("rid")
    s2 = pl.read_parquet(pq / "source2.parquet", columns=cols)
    s3 = pl.read_parquet(pq / "source3.parquet", columns=cols)
    pool = pl.concat([s2, s3])

    q = band["q"].to_numpy()
    pid = band["pid"].to_numpy()
    a_text = (s1[q]["core1"] + " " + s1[q]["addr"]).to_list()
    b_text = _pool_text(pool, pid, s2.height)

    from sentence_transformers import CrossEncoder

    model = CrossEncoder(str(mdl / "xenc"), trust_remote_code=True, max_length=128)
    xs = model.predict(list(zip(a_text, b_text)), batch_size=a.batch_size, show_progress_bar=False,
                        apply_softmax=False)
    xs = 1.0 / (1.0 + np.exp(-np.asarray(xs, dtype=np.float32).reshape(-1)))  # logit -> [0, 1]

    out = band.with_columns(pl.Series("xs", xs))
    out_dir = P["work"] / "v2" / a.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out.write_parquet(out_dir / f"xenc_{a.split}.parquet", compression="zstd")
    print(f"wrote {out_dir / f'xenc_{a.split}.parquet'}: {out.height} rows in {time.time() - t0:.0f}s", flush=True)
    log_stage("score_xenc", {"name": a.name, "split": a.split}, {"pairs": float(out.height), "seconds": time.time() - t0})


if __name__ == "__main__":
    main()
