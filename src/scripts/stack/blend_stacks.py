"""Blend stacked models that were trained on the same chunk files (other XGBoost seeds, other learners): mean of the logits.
Usage: python src/scripts/stack/blend_stacks.py NEW MODEL_A MODEL_B [MODEL_C ...]
Needs for every model: models/<m>/{holdout_pred,oof_tune}.parquet (from `stack train`) and output/<m>/pair_p.parquet (from `stack predict`). Writes models/NEW/ (holdout_pred, oof_tune,
holdout.json with the threshold tuned on the blended out-of-fold probabilities, config.json) and the outputs of NEW (pair_p and the two TSV files, validated) so that paired_models.py,
reemit.py and france_variants.py work on NEW like on any model."""

import json
import shutil
import sys
import time

import numpy as np
import polars as pl

from ber import config, decision
from ber.split import holdout_q
from ber.stages.predict import emit


def lg(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(np.float64), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def blend(frames: list[pl.DataFrame]) -> pl.DataFrame:
    fr = [f.sort("q", "pid") for f in frames]
    for f in fr[1:]:
        assert f.height == fr[0].height and (f["q"] == fr[0]["q"]).all() and (f["pid"] == fr[0]["pid"]).all(), "models must score the same pairs"
    m = np.mean([lg(f["p"].to_numpy()) for f in fr], axis=0)
    return fr[0].with_columns(pl.Series("p", (1 / (1 + np.exp(-m))).astype(np.float32)))


def main() -> None:
    new, names = sys.argv[1], sys.argv[2:]
    P = config.paths()
    t0 = time.time()
    mdl = P["work"] / "models"
    out = mdl / new
    out.mkdir(parents=True, exist_ok=True)
    oof = blend([pl.read_parquet(mdl / n / "oof_tune.parquet") for n in names])
    hold = blend([pl.read_parquet(mdl / n / "holdout_pred.parquet") for n in names])
    labels = pl.read_parquet(P["parquet"] / "train" / "labels.parquet").group_by("s1_rid").len().rename({"s1_rid": "q", "len": "n_true"})
    nt_tune = oof.select("q").unique().join(labels, on="q", how="left").with_columns(pl.col("n_true").fill_null(0))
    thr, score_oof, _ = decision.tune_threshold(oof, nt_tune, True)
    hq = pl.DataFrame({"q": holdout_q()})
    nt_h = hq.join(labels, on="q", how="left").with_columns(pl.col("n_true").fill_null(0))
    f = decision.macro_f05(decision.assign_exclusive(hold).filter(pl.col("p") >= thr), nt_h)
    print(f"blend of {names}: OOF {score_oof:.5f} at threshold {thr:.2f}; holdout {f:.5f}; single models "
          + ", ".join(f"{n} {json.loads((mdl / n / 'holdout.json').read_text())['stack_holdout_f05']:.5f}" for n in names), flush=True)
    oof.write_parquet(out / "oof_tune.parquet", compression="zstd")
    hold.write_parquet(out / "holdout_pred.parquet", compression="zstd")
    rep = json.loads((mdl / names[0] / "holdout.json").read_text())
    rep.update({"stack_holdout_f05": f, "stack_threshold": thr, "oof_subsample_f05": score_oof, "blend_of": names})
    (out / "holdout.json").write_text(json.dumps(rep, indent=2))
    cfg = json.loads((mdl / names[0] / "config.json").read_text())
    cfg.update({"threshold": thr, "blend_of": names})
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    shutil.copy(mdl / names[0] / "xgb.json", out / "xgb.json")  # placeholder so that tools looking for a model file find one
    pp = blend([pl.read_parquet(P["work"] / "output" / n / "pair_p.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64)) for n in names])
    (P["work"] / "output" / new).mkdir(parents=True, exist_ok=True)
    pp.write_parquet(P["work"] / "output" / new / "pair_p.parquet", compression="zstd")
    emit(new, P, pp, {"threshold": thr, "exclusive": True}, t0)


if __name__ == "__main__":
    main()
