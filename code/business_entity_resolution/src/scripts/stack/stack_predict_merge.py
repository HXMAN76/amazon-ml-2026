"""Score one country's rebuilt stack chunks with an existing stacked model and merge them into that model's test probabilities.
Usage: python src/scripts/stack/stack_predict_merge.py NEWNAME BASE TAG
`stack build --split test --tag TAG --ctry france ...` writes chunks for France's S1 only (for example with France's cross-encoder scores
swapped, xs_merge.py); this scores them with BASE's XGBoost model, replaces those pairs in output/BASE/pair_p.parquet (every other pair keeps
BASE's probability, so its matches are unchanged), and writes output/NEWNAME and models/NEWNAME (BASE's config: threshold, exclusivity)."""

import json
import shutil
import sys
import time

import numpy as np
import polars as pl
import xgboost as xgb

from ber import config
from ber.stages.predict import emit


def main() -> None:
    new, base, tag = sys.argv[1:4]
    P = config.paths()
    t0 = time.time()
    mdl = P["work"] / "models" / base
    cfg = json.loads((mdl / "config.json").read_text())
    model = xgb.Booster()
    model.load_model(str(mdl / "xgb.json"))
    parts = []
    for f in sorted((P["work"] / f"stack{tag}" / "test").glob("chunk_*.parquet")):
        d = pl.read_parquet(f)
        pp = model.predict(xgb.DMatrix(d.select(cfg["features"]).to_numpy().astype(np.float32), feature_names=cfg["features"]))
        parts.append(d.select(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64)).with_columns(pl.Series("p_new", pp)))
    upd = pl.concat(parts)
    old = pl.read_parquet(P["work"] / "output" / base / "pair_p.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    m = old.join(upd, on=["q", "pid"], how="left")
    print(f"{upd.height} pairs rescored of {old.height} ({m['p_new'].is_not_null().sum()} matched); mean p of those {m.filter(pl.col('p_new').is_not_null())['p'].mean():.4f} "
          f"-> {m['p_new'].mean():.4f}", flush=True)
    df = m.with_columns(pl.coalesce("p_new", "p").cast(pl.Float32).alias("p")).drop("p_new")
    (P["work"] / "models" / new).mkdir(parents=True, exist_ok=True)
    for f in ("config.json", "holdout.json", "xgb.json"):
        shutil.copy(mdl / f, P["work"] / "models" / new / f)
    (P["work"] / "output" / new).mkdir(parents=True, exist_ok=True)
    df.write_parquet(P["work"] / "output" / new / "pair_p.parquet", compression="zstd")
    emit(new, P, df, cfg, t0)


if __name__ == "__main__":
    main()
