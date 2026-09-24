"""End-to-end pipeline: data -> blocking -> pair features -> LightGBM -> threshold -> TSV outputs.

    python -m ber.run train   --data dataset --model models/v0
    python -m ber.run predict --data dataset --model models/v0 --out output
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from ber import blocking, decide, model
from ber.data import Split, load_split
from ber.features import Encoder, pair_features
from ber.metrics import breakdown
from ber.validate import validate


def build(sp: Split, k_name: int, k_both: int, country_block: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (others, pairs, features) for a split."""
    oth = sp.others
    enc = Encoder().fit(sp.s1, oth)
    e1, eo = enc.transform(sp.s1), enc.transform(oth)
    t = time.time()
    pairs = blocking.generate(sp.s1, oth, e1, eo, k_name, k_both, country_block)
    print(f"blocking: {len(pairs)} candidates ({len(pairs) / len(sp.s1):.1f}/S1) in {time.time() - t:.1f}s", flush=True)
    t = time.time()
    feats = pair_features(sp.s1, oth, pairs, e1, eo)
    print(f"features: {feats.shape[1]} cols in {time.time() - t:.1f}s", flush=True)
    return oth, pairs, feats


def cmd_train(a: argparse.Namespace) -> None:
    sp = load_split(a.data, "train")
    assert sp.truth is not None, "train ground truth missing"
    oth, pairs, feats = build(sp, a.k_name, a.k_both, not a.no_country_block)
    rec = blocking.recall(pairs, sp.s1, oth, sp.truth)
    print("blocking:", json.dumps(rec), flush=True)

    owners: dict[str, int] = {}
    for ms in sp.truth.values():
        for m in ms:
            owners[m] = owners.get(m, 0) + 1
    multi_owner = sum(v > 1 for v in owners.values())
    print(f"truth: {len(owners)} matched S2/S3 records, {multi_owner} claimed by >1 S1 entity", flush=True)

    y = model.label_pairs(pairs, sp.s1, oth, sp.truth)
    print(f"pairs: {len(y)}, positives {int(y.sum())} ({y.mean():.3%})", flush=True)
    oof, best_iters = model.cv_oof(feats, y, a.folds)
    s1_ids, oth_ids = sp.s1["entity_id"].to_numpy(), oth["entity_id"].to_numpy()
    best = decide.tune(pairs, oof, s1_ids, oth_ids, sp.truth)
    pred = decide.select(pairs, oof, s1_ids, oth_ids, best["thr"], best["one_to_one"])
    bd = breakdown(pred, sp.truth)
    print("OOF best:", {k: v for k, v in best.items() if k != "table"}, flush=True)
    print("OOF breakdown:", json.dumps(bd), flush=True)

    m = model.fit_full(feats, y, int(np.mean(best_iters) * 1.1) + 1)
    out = Path(a.model)
    out.mkdir(parents=True, exist_ok=True)
    m.save_model(str(out / "lgbm.txt"))
    cfg = {"thr": best["thr"], "one_to_one": best["one_to_one"], "k_name": a.k_name, "k_both": a.k_both,
           "country_block": not a.no_country_block}
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    (out / "report.json").write_text(json.dumps(
        {"blocking": rec, "oof": bd, "best": {k: v for k, v in best.items() if k != "table"},
         "multi_owner_records": multi_owner}, indent=2))
    imp = pd.Series(m.feature_importance("gain"), index=m.feature_name()).sort_values(ascending=False)
    print("top features:\n", imp.head(12).round(0).to_string(), flush=True)


def cmd_predict(a: argparse.Namespace) -> None:
    cfg = json.loads((Path(a.model) / "config.json").read_text())
    m = lgb.Booster(model_file=str(Path(a.model) / "lgbm.txt"))
    sp = load_split(a.data, "test")
    oth, pairs, feats = build(sp, cfg["k_name"], cfg["k_both"], cfg["country_block"])
    p = model.predict(m, feats)
    s1_ids, oth_ids = sp.s1["entity_id"].to_numpy(), oth["entity_id"].to_numpy()
    pred = decide.select(pairs, p, s1_ids, oth_ids, cfg["thr"], cfg["one_to_one"])
    cand: dict[str, list[str]] = {e: [] for e in s1_ids}
    for i, j in zip(pairs["i"].to_numpy(), pairs["j"].to_numpy()):
        cand[s1_ids[i]].append(oth_ids[j])

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"source1_entity_id": s1_ids,
                  "matched_entity_ids": [",".join(sorted(pred[e])) for e in s1_ids]}
                 ).to_csv(out / "matching_results.tsv", sep="\t", index=False)
    pd.DataFrame({"source1_entity_id": s1_ids,
                  "candidate_entity_ids": [",".join(sorted(cand[e])) for e in s1_ids]}
                 ).to_csv(out / "candidate_pairs.tsv", sep="\t", index=False)
    n_match = sum(bool(v) for v in pred.values())
    print(f"wrote {out}: {len(s1_ids)} S1, {n_match} with matches ({n_match / len(s1_ids):.1%})", flush=True)
    issues = validate(out / "matching_results.tsv", out / "candidate_pairs.tsv", Path(a.data) / "test")
    print("VALIDATION:", "PASS" if not issues else issues, flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--model", required=True)
    t.add_argument("--folds", type=int, default=5)
    t.add_argument("--k-name", type=int, default=25)
    t.add_argument("--k-both", type=int, default=25)
    t.add_argument("--no-country-block", action="store_true")
    t.set_defaults(fn=cmd_train)
    q = sub.add_parser("predict")
    q.add_argument("--data", required=True)
    q.add_argument("--model", required=True)
    q.add_argument("--out", required=True)
    q.set_defaults(fn=cmd_predict)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
