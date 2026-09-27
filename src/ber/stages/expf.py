"""expf: probability calibration and per-S1 expected-F_0.5 selection on top of the stacked probabilities.

Per S1 the metric is F_0.5 = 1.25 * TP / (0.25 * T + k) for k kept pairs, TP of them true and T true matches in all
(an empty prediction scores 1 only if T = 0). Given calibrated probabilities p_1 >= p_2 >= ... of the S1's candidates,
treated as independent, the best set to keep is a top-k prefix (Ye et al. 2012), and the expected score of every k
is computed exactly: TP among the first k and the number M of true matches among the rest are Poisson-binomial; matches the
blocker never proposed enter as a Poisson term with mean `mu`. The k with the largest expectation is kept
(k = 0 allowed, so singletons are handled by the same rule).

  python -m ber.stages.expf fit     --base s1 --name e1    calibrate on OOF, tune (mu, shift), judge on the locked holdout
  python -m ber.stages.expf predict --base s1 --name e1    apply to the test pair probabilities and write the TSVs
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import polars as pl

from ber import config, decision
from ber.stages.predict import emit
from ber.tracking import log_stage


# ------------------------------------------------------------------ calibration
def fit_isotonic(p: np.ndarray, y: np.ndarray, n_bins: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """Monotone calibration map (knots) by pool-adjacent-violators on quantile bins of the scores."""
    order = np.argsort(p, kind="stable")
    ps, ys = p[order].astype(np.float64), y[order].astype(np.float64)
    edges = np.linspace(0, len(ps), min(n_bins, len(ps)) + 1).astype(int)
    xb = np.array([ps[a:b].mean() for a, b in zip(edges[:-1], edges[1:]) if b > a])
    yb = np.array([ys[a:b].mean() for a, b in zip(edges[:-1], edges[1:]) if b > a])
    wb = np.array([b - a for a, b in zip(edges[:-1], edges[1:]) if b > a], dtype=np.float64)
    vals, wts, xs = list(yb), list(wb), list(xb)
    i = 0
    while i < len(vals) - 1:  # pool adjacent violators
        if vals[i] > vals[i + 1]:
            w = wts[i] + wts[i + 1]
            vals[i] = (vals[i] * wts[i] + vals[i + 1] * wts[i + 1]) / w
            xs[i] = (xs[i] * wts[i] + xs[i + 1] * wts[i + 1]) / w
            wts[i] = w
            del vals[i + 1], wts[i + 1], xs[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    return np.array(xs), np.array(vals)


def apply_isotonic(p: np.ndarray, knots: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Apply a calibration map; scores outside the fitted range are clamped to the end values."""
    return np.interp(p, knots[0], knots[1])


def ece(p: np.ndarray, y: np.ndarray, bins: int = 20) -> float:
    """Expected calibration error with equal-width bins."""
    idx = np.minimum((p * bins).astype(int), bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(tot)


# ------------------------------------------------------------------ expected F_0.5
def _bern(dist: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Distribution of (count + Bernoulli(p)) for each row; dist is (n, width)."""
    out = dist * (1 - p)[:, None]
    out[:, 1:] += dist[:, :-1] * p[:, None]
    return out


def expected_f05_all(P: np.ndarray, lam: np.ndarray, m_max: int = 24) -> np.ndarray:
    """Expected F_0.5 of keeping the top k of each row's probabilities, for k = 0..J. P is (n, J) sorted descending;
    lam (n,) is the Poisson mean of true matches outside the listed candidates. Returns (n, J + 1)."""
    n, J = P.shape
    D = [np.zeros((n, J + 1)) for _ in range(J + 1)]  # D[k]: distribution of TP among the first k
    D[0][:, 0] = 1.0
    for k in range(1, J + 1):
        D[k] = _bern(D[k - 1], P[:, k - 1])
    S = [None] * (J + 1)  # S[k]: distribution of true matches among candidates k..J-1
    S[J] = np.zeros((n, J + 1))
    S[J][:, 0] = 1.0
    for k in range(J - 1, -1, -1):
        S[k] = _bern(S[k + 1], P[:, k])
    pois = np.zeros((n, m_max + 1))  # Poisson pmf with the tail folded into the last bin
    pois[:, 0] = np.exp(-lam)
    for m in range(1, m_max + 1):
        pois[:, m] = pois[:, m - 1] * lam / m
    pois[:, m_max] += np.clip(1.0 - pois.sum(axis=1), 0, None)
    width = J + m_max + 1
    out = np.zeros((n, J + 1))
    tp = np.arange(J + 1)[:, None]
    mm = np.arange(width)[None, :]
    for k in range(J + 1):
        M = np.zeros((n, width))
        for j in range(J + 1):
            M[:, j: j + m_max + 1] += S[k][:, j: j + 1] * pois
        if k == 0:
            out[:, 0] = D[0][:, 0] * M[:, 0]  # F = 1 only when there is no true match at all
            continue
        f = 1.25 * tp[: k + 1] / (0.25 * (tp[: k + 1] + mm) + k)  # (k+1, width)
        out[:, k] = np.einsum("nt,tm,nm->n", D[k][:, : k + 1], f, M)
    return out


def select_topk(q: np.ndarray, p: np.ndarray, mu: float, J: int = 12, chunk: int = 150_000) -> np.ndarray:
    """Boolean mask over the input pairs: the top-k prefix per S1 that maximises the expected F_0.5."""
    df = pl.DataFrame({"q": q, "p": p, "i": np.arange(len(q))}).sort(["q", "p"], descending=[False, True])
    df = df.with_columns(pl.int_range(pl.len()).over("q").alias("r"), pl.col("q").rle_id().alias("g"))
    g = df["g"].to_numpy()
    r = df["r"].to_numpy()
    pv = df["p"].to_numpy().astype(np.float64)
    ng = int(g.max()) + 1 if len(g) else 0
    P = np.zeros((ng, J))
    m = r < J
    P[g[m], r[m]] = pv[m]
    rest = np.bincount(g[~m], weights=pv[~m], minlength=ng)
    best = np.zeros(ng, dtype=np.int64)
    for s in range(0, ng, chunk):
        e = expected_f05_all(P[s: s + chunk], rest[s: s + chunk] + mu)
        best[s: s + chunk] = e.argmax(axis=1)
    keep = r < best[g]
    mask = np.zeros(len(q), dtype=bool)
    mask[df["i"].to_numpy()[keep]] = True
    return mask


# ------------------------------------------------------------------ stage
def _prep(df: pl.DataFrame, knots, shift: float) -> np.ndarray:
    """Calibrated probabilities with an optional logit shift (the single conservatism knob)."""
    pc = np.clip(apply_isotonic(df["p"].to_numpy(), knots), 1e-6, 1 - 1e-6)
    if shift:
        pc = 1 / (1 + np.exp(-(np.log(pc / (1 - pc)) + shift)))
    return pc


def _score(df: pl.DataFrame, nt: pl.DataFrame, knots, mu: float, shift: float, J: int) -> np.ndarray:
    """Per-S1 F_0.5 (sorted by q) of the expected-F selection after exclusive assignment."""
    d = decision.assign_exclusive(df.select("q", "pid", "p", "label"))
    pc = _prep(d, knots, shift)
    mask = select_topk(d["q"].to_numpy(), pc, mu, J)
    return decision.per_entity_f05(d.filter(pl.Series(mask)), nt).sort("q")["f"].to_numpy()


def fit(base: str, name: str, prm: dict) -> None:
    """Calibrate on the out-of-fold stacked probabilities, tune (mu, shift), and judge on the locked holdout."""
    P = config.paths()
    t0 = time.time()
    mdl = P["work"] / "models" / base
    tune = pl.read_parquet(mdl / "oof_tune.parquet")
    hold = pl.read_parquet(mdl / "holdout_pred.parquet")
    labels = pl.read_parquet(P["parquet"] / "train" / "labels.parquet").group_by("s1_rid").len().rename({"s1_rid": "q", "len": "n_true"})
    knots = fit_isotonic(tune["p"].to_numpy(), tune["label"].to_numpy())
    ph = hold["p"].to_numpy()
    ece_before = ece(ph, hold["label"].to_numpy())
    ece_after = ece(apply_isotonic(ph, knots), hold["label"].to_numpy())
    print(f"holdout ECE before {ece_before:.5f} after isotonic {ece_after:.5f}", flush=True)

    nt_t = tune.select("q").unique().join(labels, on="q", how="left").with_columns(pl.col("n_true").fill_null(0))
    rng = np.random.default_rng(prm["seed"])
    sub_q = rng.choice(nt_t["q"].to_numpy(), size=min(prm["tune_q"], nt_t.height), replace=False)
    tune_s = tune.join(pl.DataFrame({"q": sub_q}), on="q", how="semi")
    nt_s = nt_t.join(pl.DataFrame({"q": sub_q}), on="q", how="semi")
    # blocking misses: true matches not among the candidates, per S1
    found = tune_s.group_by("q").agg(pl.col("label").sum().alias("found"))
    miss = nt_s.join(found, on="q", how="left").with_columns((pl.col("n_true") - pl.col("found").fill_null(0)).alias("miss"))
    mu_data = float(miss["miss"].mean())
    print(f"mean true matches missed by blocking per S1 (data estimate for mu): {mu_data:.3f}", flush=True)
    best = (-1.0, None, None)
    grid = []
    for mu in sorted(set(prm["mu_grid"] + [round(mu_data, 3)])):
        for shift in prm["shift_grid"]:
            f = float(_score(tune_s, nt_s, knots, mu, shift, prm["J"]).mean())
            grid.append((mu, shift, f))
            print(f"  tune mu={mu:.3f} shift={shift:+.2f} macro F0.5={f:.5f}", flush=True)
            if f > best[0]:
                best = (f, mu, shift)
    f_t, mu_b, shift_b = best
    print(f"best on the tuning S1: mu={mu_b} shift={shift_b} F0.5={f_t:.5f}", flush=True)

    hq = hold.select("q").unique()
    nt_h = hq.join(labels, on="q", how="left").with_columns(pl.col("n_true").fill_null(0))
    base_cfg = json.loads((mdl / "config.json").read_text())
    a_pred = decision.assign_exclusive(hold.select("q", "pid", "p", "label")).filter(pl.col("p") >= base_cfg["threshold"])
    ea = decision.per_entity_f05(a_pred, nt_h).sort("q")["f"].to_numpy()
    eb = _score(hold, nt_h, knots, mu_b, shift_b, prm["J"])
    delta, lo, hi = decision.paired_bootstrap_delta(ea, eb)
    rep = {"base": base, "base_holdout_f05_threshold_rule": float(ea.mean()), "expf_holdout_f05": float(eb.mean()), "delta": delta,
           "delta_ci95": [lo, hi], "ship": bool(lo > 0), "mu": mu_b, "shift": shift_b, "mu_data": mu_data,
           "ece_before": ece_before, "ece_after": ece_after, "holdout_s1": hq.height}
    out = P["work"] / "models" / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({"base": base, "mu": mu_b, "shift": shift_b, "J": prm["J"],
                                                 "knots_x": knots[0].tolist(), "knots_y": knots[1].tolist(),
                                                 "exclusive": False, "threshold": 0.5}))
    (out / "holdout.json").write_text(json.dumps(rep, indent=2))
    print("HOLDOUT paired comparison (expected-F0.5 selection vs threshold rule):", json.dumps(rep), flush=True)
    log_stage("expf_fit", prm, {"holdout_base": float(ea.mean()), "holdout_expf": float(eb.mean()), "delta": delta, "delta_lo": lo,
                                "seconds": time.time() - t0})


def predict(base: str, name: str) -> None:
    """Apply calibration and expected-F selection to the test pair probabilities and write the TSV outputs."""
    P = config.paths()
    t0 = time.time()
    cfg = json.loads((P["work"] / "models" / name / "config.json").read_text())
    knots = (np.array(cfg["knots_x"]), np.array(cfg["knots_y"]))
    df = pl.read_parquet(P["work"] / "output" / base / "pair_p.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    d = decision.assign_exclusive(df.with_columns(pl.lit(0).alias("label")))
    pc = _prep(d, knots, cfg["shift"])
    mask = select_topk(d["q"].to_numpy(), pc, cfg["mu"], cfg["J"])
    chosen = d.select("q", "pid").filter(pl.Series(mask)).with_columns(pl.lit(1.0).alias("sel"))
    out_df = df.join(chosen, on=["q", "pid"], how="left").with_columns(pl.col("sel").fill_null(0.0).alias("p")).select("q", "pid", "p")
    emit(name, P, out_df, {"exclusive": False, "threshold": 0.5}, t0)


def main(argv: list[str] | None = None) -> None:
    """CLI: fit | predict (see module docstring)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "predict"])
    ap.add_argument("--base", default=None)
    ap.add_argument("--name", default="e1")
    a = ap.parse_args(argv)
    prm = config.load()["expf"]
    base = a.base or prm["base"]
    fit(base, a.name, prm) if a.cmd == "fit" else predict(base, a.name)


if __name__ == "__main__":
    main()
