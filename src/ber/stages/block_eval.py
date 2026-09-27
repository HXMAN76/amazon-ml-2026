"""block_eval: recall of the candidate generator per token-type combination on a train S1 subset.

Answers, for each combination: how many true pairs are found (pair recall), how many S1 entities get all
their matches, candidates per S1, recall by country/source, and how many true pairs each channel covers.
Gate from plan.md: pair recall >= 0.98 at <= 30 candidates per S1.
"""

from __future__ import annotations

import json
import shutil
import time

import duckdb
import numpy as np
import polars as pl

from ber import config
from ber.stages import block
from ber.tracking import log_stage


def metrics(con: duckdb.DuckDBPyConnection, shard_dir, labels: str, s1: str) -> dict:
    """Pair recall, per-country/source recall, candidates per S1 and channel coverage against the labels."""
    cand = f"read_parquet('{shard_dir}/cand_*.parquet')"
    con.execute(f"""CREATE OR REPLACE TEMP TABLE tp AS
        SELECT l.s1_rid, l.src, l.src * {block.PID_BASE} + l.other_rid AS pid, s.ctry, cnt.n AS n_matches, c.pid IS NOT NULL AS found,
               {", ".join(f"c.s_{t} > 0 AS in_{t}" for t in block.TYPES)}
        FROM read_parquet('{labels}') l
        JOIN qry ON qry.q = l.s1_rid
        JOIN (SELECT rid, ctry FROM read_parquet('{s1}')) s ON s.rid = l.s1_rid
        JOIN (SELECT s1_rid, count(*) AS n FROM read_parquet('{labels}') GROUP BY 1) cnt ON cnt.s1_rid = l.s1_rid
        LEFT JOIN {cand} c ON c.q = l.s1_rid AND c.pid = l.src * {block.PID_BASE} + l.other_rid""")
    one = lambda q: con.execute(q).fetchone()[0]  # noqa: E731
    out = {
        "pair_recall": one("SELECT avg(found::INT) FROM tp"),
        "s1_all_found": one("SELECT avg(f) FROM (SELECT s1_rid, min(found::INT) AS f FROM tp GROUP BY 1)"),
        "s1_any_found": one("SELECT avg(f) FROM (SELECT s1_rid, max(found::INT) AS f FROM tp GROUP BY 1)"),
        "cand_per_s1": one(f"SELECT count(*) * 1.0 / (SELECT count(*) FROM qry) FROM {cand}"),
        "s1_without_candidates": one(f"SELECT 1 - count(DISTINCT q) * 1.0 / (SELECT count(*) FROM qry) FROM {cand}"),
    }
    for k in ("ctry", "src"):
        for key, v in con.execute(f"SELECT {k}, avg(found::INT) FROM tp GROUP BY 1").fetchall():
            out[f"recall_{k}_{key}"] = v
    for ch in block.TYPES:
        out[f"coverage_{ch}"] = one(f"SELECT avg(in_{ch}::INT) FROM tp WHERE found")
    for lo, hi in ((1, 1), (2, 3), (4, 99)):
        out[f"recall_matches_{lo}-{hi}"] = one(f"SELECT avg(found::INT) FROM tp WHERE n_matches BETWEEN {lo} AND {hi}")
    return out


def diagnose(con: duckdb.DuckDBPyConnection, labels: str) -> dict:
    """Lexical reachability of the true pairs with NO df cap: shared tokens and the rarest shared token's df."""
    con.execute(f"""CREATE OR REPLACE TEMP TABLE tpairs AS
        SELECT s1_rid AS q, src * {block.PID_BASE} + other_rid AS pid FROM read_parquet('{labels}')
        WHERE s1_rid IN (SELECT q FROM qry)""")
    con.execute("""CREATE OR REPLACE TEMP TABLE qall AS
        SELECT id AS q, typ || ':' || w AS tok FROM qbase UNION ALL SELECT id, typ || ':' || w FROM qcomp""")
    con.execute("""CREATE OR REPLACE TEMP TABLE shared AS
        SELECT t.q, t.pid, count(*) AS n_shared, min(d.df) AS min_df
        FROM tpairs t JOIN qall a ON a.q = t.q JOIN ptok_all p ON p.id = t.pid AND p.tok = a.tok
        JOIN df d ON d.tok = a.tok GROUP BY 1, 2""")
    con.execute("""CREATE OR REPLACE TEMP TABLE reach AS
        SELECT t.q, t.pid, coalesce(s.n_shared, 0) AS n_shared, s.min_df FROM tpairs t LEFT JOIN shared s USING (q, pid)""")
    one = lambda q: con.execute(q).fetchone()[0]  # noqa: E731
    out = {"true_pairs": one("SELECT count(*) FROM reach"), "no_shared_token": one("SELECT avg((n_shared = 0)::INT) FROM reach")}
    for cap in (100, 800, 5000, 20000, 100000):
        out[f"rarest_shared_df<={cap}"] = one(f"SELECT avg((min_df <= {cap})::INT) FROM reach")
    return out


def main() -> None:
    """CLI: evaluate blocking configurations on a subset of train S1 and print recall and miss breakdown."""
    P, cfg = config.paths(), config.load()
    base, ev = cfg["block"], cfg["block_eval"]
    s1 = P["sample"] / "train_s1.parquet"
    rng = np.random.default_rng(ev["seed"])
    smp = pl.read_parquet(s1)
    sub = smp[np.sort(rng.choice(smp.height, size=min(ev["n_s1"], smp.height), replace=False))].select("rid", "ctry")
    subp = P["work"] / "blocks" / "eval_s1.parquet"
    subp.parent.mkdir(parents=True, exist_ok=True)
    sub.write_parquet(subp)
    labels = str(P["parquet"] / "train" / "labels.parquet")
    con = block.open_db("train", subp, base)
    t = time.time()
    block.build_pool_index(con, base)
    print(f"pool index ready in {time.time() - t:.0f}s", flush=True)
    report = {}
    first = True
    for run in ev["runs"]:
        prm = {**base, **run["over"]}
        block.build_query_index(con, prm)
        if first:
            first = False
            d = diagnose(con, labels)
            report["diagnose"] = d
            print("[diagnose] " + json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in d.items()}), flush=True)
        name = run["name"]
        out = P["work"] / "blocks" / "eval" / name
        shutil.rmtree(out, ignore_errors=True)
        t = time.time()
        block.score_batches(con, prm, out, run["types"])
        m = metrics(con, out, labels, str(subp))
        # why are true pairs missing? unreachable / over the df cap / lost to per-type limits or top-K
        r = con.execute(f"""SELECT count(*) FILTER (WHERE NOT tp.found),
                count(*) FILTER (WHERE NOT tp.found AND r.n_shared = 0),
                count(*) FILTER (WHERE NOT tp.found AND r.n_shared > 0 AND r.min_df > {prm['cap_df']}),
                count(*) FILTER (WHERE NOT tp.found AND r.n_shared > 0 AND r.min_df <= {prm['cap_df']})
            FROM tp JOIN reach r ON r.q = tp.s1_rid AND r.pid = tp.pid""").fetchone()
        n = max(con.execute("SELECT count(*) FROM tp").fetchone()[0], 1)
        m.update({"miss_unreachable": r[1] / n, "miss_over_cap": r[2] / n, "miss_truncated": r[3] / n})
        m["seconds"] = time.time() - t
        report[name] = m
        by_ctry = {k[len("recall_ctry_"):]: round(v, 4) for k, v in m.items() if k.startswith("recall_ctry_")}
        cov = ",".join(f"{ch}={m[f'coverage_{ch}']:.2f}" for ch in block.TYPES)
        print(f"[{name}] recall={m['pair_recall']:.4f} all_found={m['s1_all_found']:.4f} cand/S1={m['cand_per_s1']:.1f} "
              f"no_cand={m['s1_without_candidates']:.4f} by_ctry={by_ctry} misses(unreach/cap/trunc)="
              f"{m['miss_unreachable']:.3f}/{m['miss_over_cap']:.3f}/{m['miss_truncated']:.3f} cov[{cov}] {m['seconds']:.0f}s", flush=True)
        log_stage(f"block_eval_{name}", {"run": name, "over": run["over"], "types": run["types"]},
                  {k: float(v) for k, v in m.items()})
    (P["work"] / "blocks" / "eval_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
