"""setmodel: an entity-level model that scores all shortlist candidates of one S1 together, then a blend with the XGBoost stack.

The stack sees each (S1, candidate) row with hand-made summaries of the S1's other candidates (counts, best other record, digit
agreement). Look-alike decoys are told apart only by comparing a candidate with the S1's confirmed records, so here every S1 is one
sequence of its shortlist candidates (<= shortlist_k), each token carrying the stack's features and the stack's out-of-fold
probability, and a small transformer lets every candidate attend to its siblings before it is scored. Same rows, same S1 hash folds
and the same locked holdout as the stack, so the two are directly comparable and can be blended.

  python -m ber.stages.setmodel train --name bs_set --stack bs_w2
        -> models/bs_set/{oof,holdout_pred}.parquet (q, pid, label, p), output/bs_set/pair_p.parquet (q, pid, p)
  python -m ber.stages.setmodel blend --name bs_final --stack bs_w2 --set bs_set
        -> blend weight and threshold tuned out-of-fold, paired bootstrap against the stack alone on the locked holdout,
           models/bs_final/{config,holdout}.json; output/bs_final/ TSVs through predict.emit
Needs torch (requirements-gpu.txt); runs on CPU for the tests.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import polars as pl

from ber import config, decision
from ber.split import final_q, holdout_q
from ber.tracking import log_stage

PRM = {"d": 128, "layers": 3, "heads": 4, "epochs": 4, "batch": 1024, "lr": 1e-3, "folds": 5, "seed": 0}


def hash_fold(q: np.ndarray, k: int) -> np.ndarray:
    """The stack's S1 fold (stack.train): the same S1 land in the same fold."""
    u = q.astype(np.uint64)
    return ((u * np.uint64(2654435761)) % np.uint64(2 ** 32) % np.uint64(k)).astype(np.int64)


def load(stack: str, split: str) -> tuple[pl.DataFrame, list[str]]:
    """Stack feature rows of a split (q, pid, label, features, p_stack), sorted by q: p_stack is out-of-fold on the stack's training
    S1, the stack's own prediction on the holdout and on test."""
    P = config.paths()
    cfg = json.loads((P["work"] / "models" / stack / "config.json").read_text())
    feats = cfg["features"]
    folder = P["work"] / ("stack" + cfg.get("tag", "")) / split
    cols = ["q", "pid", *(["label"] if split == "train" else []), *feats]
    d = pl.concat([pl.read_parquet(f, columns=cols) for f in sorted(folder.glob("chunk_*.parquet"))]).with_columns(
        pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    if split == "train":  # out-of-fold on the training S1, the stack's own p on the locked and the final holdout
        M = P["work"] / "models" / stack
        ps = pl.concat([pl.read_parquet(M / f, columns=["q", "pid", "p"]) for f in ("oof_tune.parquet", "holdout_pred.parquet", "final_pred.parquet")
                        if (M / f).exists()])
    else:
        ps = pl.read_parquet(P["work"] / "output" / stack / "pair_p.parquet", columns=["q", "pid", "p"])
        d = d.with_columns(pl.lit(0, dtype=pl.Int8).alias("label"))
    ps = ps.select(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64), pl.col("p").alias("p_stack"))
    return d.join(ps, on=["q", "pid"], how="inner").sort("q", "pid"), feats


def matrix(d: pl.DataFrame, feats: list[str], stats: dict | None = None) -> tuple[np.ndarray, dict]:
    """Standardised float32 inputs: features, logit of p_stack, and a missing indicator for each column that can be missing."""
    x = d.select(feats).to_numpy().astype(np.float32)
    p = np.clip(d["p_stack"].to_numpy().astype(np.float32), 1e-6, 1 - 1e-6)
    x = np.concatenate([x, np.log(p / (1 - p))[:, None]], axis=1)
    if stats is None:
        stats = {"mu": np.nanmean(x, 0), "sd": np.nanstd(x, 0) + 1e-6, "nan_cols": np.where(np.isnan(x).any(0))[0]}
    miss = np.isnan(x[:, stats["nan_cols"]]).astype(np.float32)
    z = np.nan_to_num((x - stats["mu"]) / stats["sd"], nan=0.0)
    return np.concatenate([z, miss], axis=1).astype(np.float32), stats


class Groups:
    """Rows sorted by q, viewed as one group per S1."""

    def __init__(self, x: np.ndarray, q: np.ndarray, y: np.ndarray | None = None):
        self.x, self.y = x, (np.zeros(len(x), dtype=np.float32) if y is None else y)
        self.q, self.start, self.cnt = np.unique(q, return_index=True, return_counts=True)

    def rows(self, gs: np.ndarray) -> np.ndarray:
        """Row indices of groups gs (in group order)."""
        return np.concatenate([np.arange(self.start[g], self.start[g] + self.cnt[g]) for g in gs]) if len(gs) else np.empty(0, dtype=np.int64)

    def padded(self, gs: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(x [B, k, F], y [B, k], pad [B, k]) for groups gs; pad is True where there is no candidate."""
        B = len(gs)
        xb = np.zeros((B, k, self.x.shape[1]), dtype=np.float32)
        yb = np.zeros((B, k), dtype=np.float32)
        pad = np.ones((B, k), dtype=bool)
        for j, g in enumerate(gs):
            s, c = self.start[g], self.cnt[g]
            xb[j, :c], yb[j, :c], pad[j, :c] = self.x[s: s + c], self.y[s: s + c], False
        return xb, yb, pad


def net(f: int):
    from torch import nn

    class SetNet(nn.Module):
        def __init__(self):
            super().__init__()
            d = PRM["d"]
            self.inp = nn.Sequential(nn.Linear(f, d), nn.GELU(), nn.Linear(d, d))
            layer = nn.TransformerEncoderLayer(d, PRM["heads"], 2 * d, dropout=0.1, batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, PRM["layers"], enable_nested_tensor=False)
            self.out = nn.Linear(d, 1)

        def forward(self, x, pad):
            return self.out(self.enc(self.inp(x), src_key_padding_mask=pad)).squeeze(-1)

    return SetNet()


def fit_predict(G: Groups, tr_g: np.ndarray, pred: list[tuple[Groups, np.ndarray]], k: int, device: str, seed: int) -> list[np.ndarray]:
    """Train on groups tr_g of G (a tenth kept aside to pick the best epoch); per-row probabilities for each (groups, ids) in pred."""
    import torch

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tr_g = rng.permutation(tr_g)
    n_val = max(1, len(tr_g) // 10)
    val_g, fit_g = tr_g[:n_val], tr_g[n_val:]
    model = net(G.x.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=PRM["lr"], weight_decay=0.01)
    lossf = torch.nn.BCEWithLogitsLoss(reduction="none")

    def epoch(gs: np.ndarray, train: bool) -> float:
        model.train(train)
        tot, n = 0.0, 0
        for a in range(0, len(gs), PRM["batch"]):
            xb, yb, pb = (torch.from_numpy(t).to(device) for t in G.padded(gs[a: a + PRM["batch"]], k))
            with torch.set_grad_enabled(train):
                live = (~pb).float()
                loss = (lossf(model(xb, pb), yb) * live).sum() / live.sum()
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            tot, n = tot + float(loss) * len(xb), n + len(xb)
        return tot / max(n, 1)

    best, state = np.inf, None
    for ep in range(PRM["epochs"]):
        tl = epoch(rng.permutation(fit_g), True)
        vl = epoch(val_g, False)
        print(f"  epoch {ep}: train loss {tl:.5f}, val loss {vl:.5f}", flush=True)
        if vl < best:
            best, state = vl, {n: t.detach().clone() for n, t in model.state_dict().items()}
    model.load_state_dict(state)
    model.eval()
    outs = []
    for H, gs in pred:
        p = np.zeros(len(H.x), dtype=np.float32)
        with torch.no_grad():
            for a in range(0, len(gs), 4 * PRM["batch"]):
                gi = gs[a: a + 4 * PRM["batch"]]
                xb, _, pb = H.padded(gi, k)
                pr = torch.sigmoid(model(torch.from_numpy(xb).to(device), torch.from_numpy(pb).to(device))).cpu().numpy()
                for j, g in enumerate(gi):
                    p[H.start[g]: H.start[g] + H.cnt[g]] = pr[j, : H.cnt[g]]
        outs.append(p)
    return outs


def train(name: str, stack: str) -> None:
    """5 grouped folds on the stack's training S1 (out-of-fold p); the fold models are averaged on the holdout and on test."""
    import torch

    P = config.paths()
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d, feats = load(stack, "train")
    x, stats = matrix(d, feats)
    G = Groups(x, d["q"].to_numpy(), d["label"].to_numpy().astype(np.float32))
    dt, _ = load(stack, "test")
    T = Groups(matrix(dt, feats, stats)[0], dt["q"].to_numpy())
    k = int(max(G.cnt.max(), T.cnt.max()))
    is_hold = np.isin(G.q, holdout_q())
    is_final = np.isin(G.q, final_q())  # never trained on; predicted for the freeze only
    fold = hash_fold(G.q, PRM["folds"])
    print(f"{len(G.q)} train S1 ({int(is_hold.sum())} holdout), {len(T.q)} test S1, up to {k} candidates, {x.shape[1]} inputs, {device}", flush=True)
    oof = np.zeros(len(x), dtype=np.float32)
    hold_p = np.zeros(len(x), dtype=np.float32)
    test_p = np.zeros(len(T.x), dtype=np.float32)
    trainable = ~is_hold & ~is_final
    held_g, all_t = np.where(~trainable)[0], np.arange(len(T.q))
    for f in range(PRM["folds"]):
        tr_g, va_g = np.where(trainable & (fold != f))[0], np.where(trainable & (fold == f))[0]
        print(f"fold {f}: fit on {len(tr_g)} S1, out-of-fold {len(va_g)}", flush=True)
        pv, ph, pt = fit_predict(G, tr_g, [(G, va_g), (G, held_g), (T, all_t)], k, device, PRM["seed"] + f)
        rv = G.rows(va_g)
        oof[rv] = pv[rv]
        hold_p += ph / PRM["folds"]
        test_p += pt / PRM["folds"]
    row_hold, row_final = np.repeat(is_hold, G.cnt), np.repeat(is_final, G.cnt)
    row_train = ~row_hold & ~row_final
    out = P["work"] / "models" / name
    out.mkdir(parents=True, exist_ok=True)
    keys = d.select("q", "pid", "label")
    keys.filter(pl.Series(row_train)).with_columns(pl.Series("p", oof[row_train])).write_parquet(out / "oof.parquet", compression="zstd")
    keys.filter(pl.Series(row_hold)).with_columns(pl.Series("p", hold_p[row_hold])).write_parquet(out / "holdout_pred.parquet", compression="zstd")
    if row_final.any():
        keys.filter(pl.Series(row_final)).with_columns(pl.Series("p", hold_p[row_final])).write_parquet(out / "final_pred.parquet", compression="zstd")
    (P["work"] / "output" / name).mkdir(parents=True, exist_ok=True)
    dt.select("q", "pid").with_columns(pl.Series("p", test_p)).write_parquet(P["work"] / "output" / name / "pair_p.parquet", compression="zstd")
    (out / "config.json").write_text(json.dumps({"stack": stack, "params": PRM, "inputs": int(x.shape[1])}, indent=2))
    print(f"set model {name} done in {time.time() - t0:.0f}s", flush=True)
    log_stage("setmodel_train", PRM, {"seconds": time.time() - t0})


def _f05(df: pl.DataFrame, thr: float, nt: pl.DataFrame, cap: dict) -> np.ndarray:
    sel = decision.assign_exclusive(df).filter(pl.col("p") >= thr)
    if cap:
        sel = decision.cap_per_source(sel, cap)
    return decision.per_entity_f05(sel, nt).sort("q")["f"].to_numpy()


def blend(name: str, stack: str, setm: str) -> None:
    """p = a p_stack + (1 - a) p_set with a and the threshold tuned out-of-fold; shipped only if it beats the stack on the holdout."""
    from ber.stages.predict import emit

    P = config.paths()
    t0 = time.time()
    M = P["work"] / "models"
    scfg = json.loads((M / stack / "config.json").read_text())
    cap = {int(k): int(v) for k, v in (scfg.get("cap") or {}).items()}
    labels = pl.read_parquet(P["parquet"] / "train" / "labels.parquet").group_by("s1_rid").len().rename({"s1_rid": "q", "len": "n_true"}).with_columns(pl.col("q").cast(pl.Int64))
    ids = lambda f: pl.read_parquet(f).with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))  # noqa: E731
    tune = ids(M / stack / "oof_tune.parquet").rename({"p": "ps"}).join(ids(M / setm / "oof.parquet").select("q", "pid", pl.col("p").alias("pn")), on=["q", "pid"])
    hold = ids(M / stack / "holdout_pred.parquet").rename({"p": "ps"}).join(ids(M / setm / "holdout_pred.parquet").select("q", "pid", pl.col("p").alias("pn")), on=["q", "pid"])
    nt_tune = tune.select("q").unique().join(labels, on="q", how="left").with_columns(pl.col("n_true").fill_null(0))
    best = (-1.0, 1.0, scfg["threshold"])
    for a in (1.0, 0.75, 0.5, 0.25, 0.0):
        thr, f, _ = decision.tune_threshold(tune.with_columns((a * pl.col("ps") + (1 - a) * pl.col("pn")).alias("p")), nt_tune, True)
        print(f"weight on the stack {a:.2f}: out-of-fold F0.5 {f:.5f} at threshold {thr:.2f}", flush=True)
        if f > best[0]:
            best = (f, a, thr)
    f_oof, a, thr = best
    nt_h = pl.DataFrame({"q": holdout_q().astype(np.int64)}).join(labels, on="q", how="left").with_columns(pl.col("n_true").fill_null(0))
    ea = _f05(hold.with_columns(pl.col("ps").alias("p")), scfg["threshold"], nt_h, cap)
    eb = _f05(hold.with_columns((a * pl.col("ps") + (1 - a) * pl.col("pn")).alias("p")), thr, nt_h, cap)
    delta, lo, hi = decision.paired_bootstrap_delta(ea, eb)
    rep = {"stack": stack, "set": setm, "weight_stack": a, "threshold": thr, "oof_f05": f_oof, "stack_holdout_f05": float(ea.mean()),
           "blend_holdout_f05": float(eb.mean()), "delta": delta, "delta_ci95": [lo, hi], "ship": bool(lo > 0)}
    print("HOLDOUT blend vs stack:", json.dumps(rep), flush=True)
    out = M / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "holdout.json").write_text(json.dumps(rep, indent=2))
    if not rep["ship"]:  # keep the stack's own decision: same probabilities, threshold and cap
        a, thr = 1.0, scfg["threshold"]
        print("blend not better on the holdout: the output is the stack's", flush=True)
    cfg = {"threshold": thr, "exclusive": True, "cap": cap, "weight_stack": a, "stack": stack, "set": setm}
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    if (M / stack / "final_pred.parquet").exists() and (M / setm / "final_pred.parquet").exists():  # for the freeze only
        fin = ids(M / stack / "final_pred.parquet").rename({"p": "ps"}).join(ids(M / setm / "final_pred.parquet").select("q", "pid", pl.col("p").alias("pn")), on=["q", "pid"])
        fin.select("q", "pid", "label", (a * pl.col("ps") + (1 - a) * pl.col("pn")).alias("p")).write_parquet(out / "final_pred.parquet", compression="zstd")
    ps = ids(P["work"] / "output" / stack / "pair_p.parquet").rename({"p": "ps"})
    pn = ids(P["work"] / "output" / setm / "pair_p.parquet").rename({"p": "pn"})
    df = ps.join(pn, on=["q", "pid"], how="left").with_columns(pl.col("pn").fill_null(pl.col("ps"))).select(
        "q", "pid", (a * pl.col("ps") + (1 - a) * pl.col("pn")).alias("p"))
    (P["work"] / "output" / name).mkdir(parents=True, exist_ok=True)
    df.write_parquet(P["work"] / "output" / name / "pair_p.parquet", compression="zstd")
    emit(name, P, df, cfg, t0)
    log_stage("setmodel_blend", {"stack": stack, "set": setm}, {k: v for k, v in rep.items() if isinstance(v, float)})


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "blend"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--stack", required=True)
    ap.add_argument("--set", default=None, help="blend: the set model")
    ap.add_argument("--epochs", type=int, default=None)
    a = ap.parse_args(argv)
    if a.epochs:
        PRM["epochs"] = a.epochs
    if a.cmd == "train":
        train(a.name, a.stack)
    else:
        blend(a.name, a.stack, a.set)


if __name__ == "__main__":
    main()
