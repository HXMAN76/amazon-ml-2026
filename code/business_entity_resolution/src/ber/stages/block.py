"""block: candidate generation with a weighted token index in DuckDB.

Every record becomes tagged tokens:
  n  name word (core name plus alias/domain label)       a  address word or number
  p  5-character prefix of a long word (suffix typos)
  c  composite: rare name word | rare address word        m  composite: two rare name words
  d  composite: two rare address words                    h  composite: house number | rare address word
  g  whole core name with spaces removed (glued names)    x  one-deletion variants of the 2 rarest name words (typos)
  k  consonant skeleton of name words (pool: romanised non-Latin names; query: Latin names) - bridges scripts
Tokens whose document frequency in the S2+S3 pool exceeds `cap_df` are dropped. A pool record's score for an
S1 query is the sum of IDF over shared tokens; the top K per query are kept. Queries run in batches and are
written as Parquet shards. The pool-side index is cached in a persistent DuckDB file keyed by a hash of the
index params, so tuning cap_df / per_type / k does not rebuild it.

Output: WORK/blocks/{split}/cand_XXXX.parquet with
  q (S1 rid), pid (src*10_000_000 + rid), score, ns (shared tokens), s_<type> (score per token type)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import duckdb

from ber import config
from ber.tracking import log_stage

PID_BASE = 10_000_000
TYPES = ("n", "a", "p", "c", "m", "d", "h", "g", "x", "k")
INDEX_VERSION = 2  # bump when token generation changes so the cached pool index is rebuilt


def base_tokens(tbl: str, idcol: str, prefix_len: int, min_prefix: int, k_expr: str, k_where: str) -> str:
    return f"""
    SELECT DISTINCT id, typ, w FROM (
      SELECT {idcol} AS id, 'n' AS typ, unnest(string_split(core1 || ' ' || name2, ' ')) AS w FROM {tbl}
      UNION ALL
      SELECT {idcol} AS id, 'a' AS typ, unnest(string_split(addr, ' ')) AS w FROM {tbl}
    ) WHERE length(w) >= 2
    UNION
    SELECT DISTINCT id, 'p', substr(w, 1, {prefix_len}) FROM (
      SELECT {idcol} AS id, unnest(string_split(core1 || ' ' || name2 || ' ' || addr, ' ')) AS w FROM {tbl}
    ) WHERE length(w) >= {min_prefix} AND NOT regexp_matches(w, '^[0-9]+$')
    UNION
    SELECT DISTINCT {idcol}, 'g', replace(core1, ' ', '') FROM {tbl} WHERE length(replace(core1, ' ', '')) >= 6
    UNION
    SELECT DISTINCT id, 'k', regexp_replace(w, '[aeiouy]', '', 'g') FROM (
      SELECT {idcol} AS id, unnest(string_split({k_expr}, ' ')) AS w FROM {tbl} {k_where}
    ) WHERE length(w) >= 3 AND length(regexp_replace(w, '[aeiouy]', '', 'g')) >= 2
    """


def make_composites(con: duckdb.DuckDBPyConnection, bt: str, dft: str, out: str, prm: dict) -> None:
    """Composite tokens from the rarest words of each record (rarity = pool document frequency)."""
    k = prm["comp"]
    con.execute(f"""CREATE OR REPLACE TEMP TABLE rk_n AS
        SELECT b.id, b.w, row_number() OVER (PARTITION BY b.id ORDER BY d.df, b.w) AS rk
        FROM {bt} b JOIN {dft} d ON d.tok = 'n:' || b.w WHERE b.typ = 'n'""")
    con.execute(f"""CREATE OR REPLACE TEMP TABLE rk_a AS
        SELECT b.id, b.w, row_number() OVER (PARTITION BY b.id ORDER BY d.df, b.w) AS rk
        FROM {bt} b JOIN {dft} d ON d.tok = 'a:' || b.w WHERE b.typ = 'a'""")
    con.execute(f"""CREATE OR REPLACE TEMP TABLE rk_aw AS
        SELECT b.id, b.w, row_number() OVER (PARTITION BY b.id ORDER BY d.df, b.w) AS rk
        FROM {bt} b JOIN {dft} d ON d.tok = 'a:' || b.w WHERE b.typ = 'a' AND NOT regexp_matches(b.w, '[0-9]')""")
    con.execute(f"""CREATE OR REPLACE TABLE {out} AS
        SELECT n.id, 'c' AS typ, n.w || '|' || a.w AS w FROM rk_n n JOIN rk_a a ON n.id = a.id
          AND n.rk <= {k['c'][0]} AND a.rk <= {k['c'][1]}
        UNION ALL
        SELECT x.id, 'm', x.w || '|' || y.w FROM rk_n x JOIN rk_n y ON x.id = y.id AND x.w < y.w
          AND x.rk <= {k['m']} AND y.rk <= {k['m']}
        UNION ALL
        SELECT x.id, 'd', x.w || '|' || y.w FROM rk_a x JOIN rk_a y ON x.id = y.id AND x.w < y.w
          AND x.rk <= {k['d']} AND y.rk <= {k['d']}
        UNION ALL
        SELECT b.id, 'h', b.w || '|' || r.w FROM {bt} b JOIN rk_aw r ON r.id = b.id AND r.rk <= {k['h']}
          WHERE b.typ = 'a' AND regexp_matches(b.w, '[0-9]')
        UNION ALL
        SELECT DISTINCT id, 'x', v FROM (
          SELECT id, unnest(list_concat([w], list_transform(range(1, length(w) + 1),
                                                          i -> substr(w, 1, i - 1) || substr(w, i + 1)))) AS v
          FROM rk_n WHERE rk <= 2 AND length(w) BETWEEN 4 AND 12
        )""")


def index_hash(con: duckdb.DuckDBPyConnection, prm: dict) -> str:
    n = con.execute("SELECT count(*), sum(length(core1) + length(addr)) FROM pool").fetchone()
    key = json.dumps([INDEX_VERSION, prm["prefix_len"], prm["min_prefix_len"], prm["comp"], list(n)], sort_keys=True)
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def build_pool_index(con: duckdb.DuckDBPyConnection, prm: dict) -> None:
    h = index_hash(con, prm)
    try:
        if con.execute("SELECT h FROM meta").fetchone()[0] == h:
            print("pool index: cached", flush=True)
            return
    except duckdb.Error:
        pass
    t = time.time()
    for tb in ("pbase", "dfb", "pcomp", "ptok_all", "df"):
        con.execute(f"DROP TABLE IF EXISTS {tb}")
    con.execute(f"CREATE TABLE pbase AS {base_tokens('pool', 'pid', prm['prefix_len'], prm['min_prefix_len'], 'core_rom', 'WHERE nl')}")
    con.execute("CREATE TABLE dfb AS SELECT typ || ':' || w AS tok, count(*) AS df FROM pbase GROUP BY 1")
    make_composites(con, "pbase", "dfb", "pcomp", prm)
    con.execute("CREATE TABLE ptok_all AS SELECT id, typ, typ || ':' || w AS tok FROM pbase UNION ALL "
                "SELECT id, typ, typ || ':' || w FROM pcomp")
    con.execute("CREATE TABLE df AS SELECT tok, typ, count(*) AS df FROM ptok_all GROUP BY 1, 2")
    con.execute(f"CREATE OR REPLACE TABLE meta AS SELECT '{h}' AS h")
    print(f"pool index built in {time.time() - t:.0f}s", flush=True)


def build_query_index(con: duckdb.DuckDBPyConnection, prm: dict) -> dict:
    """Creates qtok (kept query tokens with idf) and ptok (pool postings restricted to those tokens)."""
    t = time.time()
    n_pool = con.execute("SELECT count(*) FROM pool").fetchone()[0]
    con.execute("DROP TABLE IF EXISTS dfk")
    con.execute(f"CREATE TABLE dfk AS SELECT tok, typ, df, ln({n_pool}::DOUBLE / df) AS idf FROM df "
                f"WHERE df <= {prm['cap_df']}")
    for tb in ("qbase", "qcomp", "qtok", "ptok"):
        con.execute(f"DROP TABLE IF EXISTS {tb}")
    con.execute(f"CREATE TABLE qbase AS {base_tokens('qry', 'q', prm['prefix_len'], prm['min_prefix_len'], 'core1', '')}")
    make_composites(con, "qbase", "dfb", "qcomp", prm)
    lim = " ".join(f"WHEN '{k}' THEN {v}" for k, v in prm["per_type"].items())
    con.execute(f"""
      CREATE TABLE qtok AS
      SELECT id AS q, typ, tok, idf FROM (
        SELECT x.id, x.typ, x.tok, d.idf,
               row_number() OVER (PARTITION BY x.id, x.typ ORDER BY d.df, x.tok) AS rk
        FROM (SELECT id, typ, typ || ':' || w AS tok FROM qbase UNION ALL
              SELECT id, typ, typ || ':' || w FROM qcomp) x
        JOIN dfk d ON d.tok = x.tok
      ) WHERE rk <= CASE typ {lim} ELSE 0 END""")
    con.execute("CREATE TABLE ptok AS SELECT p.id AS pid, p.typ, p.tok FROM ptok_all p "
                "WHERE p.tok IN (SELECT DISTINCT tok FROM qtok)")
    st = {"query_tokens": con.execute("SELECT count(*) FROM qtok").fetchone()[0],
          "pool_postings_used": con.execute("SELECT count(*) FROM ptok").fetchone()[0]}
    print(f"query index: {st} in {time.time() - t:.0f}s", flush=True)
    return st


def build_index(con: duckdb.DuckDBPyConnection, prm: dict) -> dict:
    build_pool_index(con, prm)
    return build_query_index(con, prm)


def score_batches(con: duckdb.DuckDBPyConnection, prm: dict, out: Path, types: list[str]) -> int:
    out.mkdir(parents=True, exist_ok=True)
    qs = [r[0] for r in con.execute("SELECT DISTINCT q FROM qtok ORDER BY q").fetchall()]
    total = 0
    tl = ",".join(repr(x) for x in types)
    per_type = ", ".join(f"coalesce(sum(q.idf) FILTER (WHERE q.typ = '{t}'), 0) AS s_{t}" for t in TYPES)
    cols = ", ".join(f"s_{t}" for t in TYPES)
    for bi, s in enumerate(range(0, len(qs), prm["batch_q"])):
        lo, hi = qs[s], qs[min(s + prm["batch_q"], len(qs)) - 1]
        f = out / f"cand_{bi:04d}.parquet"
        if f.exists():
            continue
        t = time.time()
        con.execute(f"""
          COPY (
            SELECT q, pid, score, ns, {cols} FROM (
              SELECT q.q AS q, p.pid AS pid, sum(q.idf) AS score, count(*) AS ns, {per_type}
              FROM qtok q JOIN ptok p ON p.tok = q.tok
              WHERE q.q BETWEEN {lo} AND {hi} AND q.typ IN ({tl})
              GROUP BY 1, 2
            ) QUALIFY row_number() OVER (PARTITION BY q ORDER BY score DESC, pid) <= {prm['k']}
          ) TO '{f}.tmp' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        Path(f"{f}.tmp").rename(f)
        n = con.execute(f"SELECT count(*) FROM read_parquet('{f}')").fetchone()[0]
        total += n
        print(f"batch {bi}: q {lo}..{hi}, {n} pairs in {time.time() - t:.0f}s", flush=True)
    return total


def open_db(split: str, qids_parquet: Path | None, prm: dict) -> duckdb.DuckDBPyConnection:
    """Persistent connection with `pool` (S2+S3) and `qry` (S1, optionally limited to rids in a Parquet)."""
    P = config.paths()
    pq = P["parquet"] / split
    (P["work"] / "blocks").mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(P["work"] / "blocks" / f"index_{split}.duckdb"))
    tmp = P["work"] / "duckdb_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    con.execute(f"PRAGMA memory_limit='{prm['memory_limit']}'; PRAGMA threads={prm['threads']}; "
                f"SET temp_directory='{tmp}'")
    con.execute(f"""CREATE OR REPLACE TABLE pool AS
        SELECT ({2 * PID_BASE} + rid)::BIGINT AS pid, core1, name2, addr, core_rom, nl_name > 0.5 AS nl FROM read_parquet('{pq}/source2.parquet')
        UNION ALL
        SELECT ({3 * PID_BASE} + rid)::BIGINT, core1, name2, addr, core_rom, nl_name > 0.5 FROM read_parquet('{pq}/source3.parquet')""")
    sel = f"WHERE rid IN (SELECT rid FROM read_parquet('{qids_parquet}'))" if qids_parquet else ""
    con.execute(f"CREATE OR REPLACE TABLE qry AS SELECT rid AS q, core1, name2, addr FROM read_parquet('{pq}/source1.parquet') {sel}")
    return con


def run(split: str, qids_parquet: Path | None, types: list[str], prm: dict, out: Path) -> dict:
    con = open_db(split, qids_parquet, prm)
    st = build_index(con, prm)
    st["pairs"] = score_batches(con, prm, out, types)
    con.close()
    return st


def params_hash(prm: dict, types: list[str], q: Path | None) -> str:
    return hashlib.sha256(json.dumps([prm, types, str(q)], sort_keys=True).encode()).hexdigest()[:12]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--all-train", action="store_true", help="train: block all S1, not just the sample")
    a = ap.parse_args(argv)
    P, prm = config.paths(), config.load()["block"]
    q = None
    if a.split == "train" and not a.all_train:
        q = P["sample"] / "train_s1.parquet"
    out = P["work"] / "blocks" / a.split
    h = params_hash(prm, prm["types"], q)
    marker = out / ".params"
    if out.exists() and (not marker.exists() or marker.read_text() != h):
        shutil.rmtree(out)  # params changed: stale shards must not be mixed with new ones
    out.mkdir(parents=True, exist_ok=True)
    marker.write_text(h)
    t = time.time()
    st = run(a.split, q, prm["types"], prm, out)
    print(f"done {a.split}: {st['pairs']} candidate pairs in {time.time() - t:.0f}s", flush=True)
    log_stage(f"block_{a.split}", prm, {**{k: float(v) for k, v in st.items()}, "seconds": time.time() - t})


if __name__ == "__main__":
    main()
