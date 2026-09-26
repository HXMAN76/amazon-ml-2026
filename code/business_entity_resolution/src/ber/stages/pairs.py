"""pairs: turn blocking candidates into a feature table for the matcher (vectorised, streamed in chunks).

Inputs : WORK/blocks/{split}/cand_*.parquet, WORK/parquet/{split}/source{1,2,3}.parquet
Outputs: WORK/features/{split}/part_XXXX.parquet  with q, pid, [label], features (see FEATURES below)

Blocking-score features are computed over the full candidate set of the split (all S1 that were blocked), so
the competition statistics (how many S1 want this S2/S3 record, how strong the best competitor is) match
test-time conditions. String features use rapidfuzz `cpdist` (multi-threaded C++) plus polars list ops.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein

from ber import config
from ber.stages.block import PID_BASE, TYPES
from ber.tracking import log_stage


def blocking_features(cand_glob: str, drop_q: np.ndarray | None = None) -> pl.DataFrame:
    """Per-pair blocking statistics: score ranks/gaps within the S1 and within the candidate record.
    `drop_q`: S1 removed from the universe before any statistic is computed (test-like universe, see `pairs.drop_frac`)."""
    c = pl.read_parquet(cand_glob)
    if drop_q is not None and len(drop_q):
        c = c.filter(~pl.col("q").is_in(drop_q))
    if "p_block" in c.columns:  # the pruner was fit on the training S1: never let its score become a model feature
        c = c.drop("p_block")
    f32 = [pl.col(x).cast(pl.Float32) for x in ("score", *[f"s_{t}" for t in TYPES])]
    c = c.with_columns(f32).with_columns(pl.col("ns").cast(pl.Int16))
    c = c.with_columns(
        pl.col("score").rank("ordinal", descending=True).over("q").cast(pl.Int16).alias("rank_q"),
        (pl.col("score").max().over("q") - pl.col("score")).alias("gap_q"),
        pl.len().over("q").cast(pl.Int16).alias("n_cand_q"),
        pl.col("score").rank("ordinal", descending=True).over("pid").cast(pl.Int16).alias("rank_p"),
        pl.len().over("pid").cast(pl.Int16).alias("n_q_for_p"),
    )
    top2 = c.group_by("pid").agg(pl.col("score").sort(descending=True).head(2).alias("t")).with_columns(
        pl.col("t").list.get(0).cast(pl.Float32).alias("p_top1"),
        pl.col("t").list.get(1, null_on_oob=True).cast(pl.Float32).alias("p_top2"),
    ).drop("t")
    c = c.join(top2, on="pid", how="left").with_columns(
        # margin to the best competing S1 for this record: positive when this S1 is the leader
        pl.when(pl.col("rank_p") == 1).then(pl.col("score") - pl.col("p_top2").fill_null(0))
          .otherwise(pl.col("score") - pl.col("p_top1")).alias("margin_p")
    ).drop("p_top1", "p_top2")
    return c


def _cp(a: list[str], b: list[str], scorer) -> np.ndarray:
    if not a:
        return np.zeros(0, dtype=np.float32)
    return process.cpdist(a, b, scorer=scorer, dtype=np.float32, workers=-1)


def skeleton(col: pl.Series) -> list[str]:
    """Consonant skeleton of Latin strings (spaces and vowels removed): a crude phonetic key that survives vowel
    differences between a Latin name and the romanisation of the same name written in an Indic script."""
    return col.str.replace_all(" ", "", literal=True).str.replace_all("[aeiouy]", "").to_list()


def _coverage(a: list[str], b: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Fraction of a's tokens found in b and of b's tokens found in a."""
    d = pl.DataFrame({"a": a, "b": b}).with_columns(pl.col("a").str.split(" ").alias("at"), pl.col("b").str.split(" ").alias("bt"))
    d = d.with_columns(pl.col("at").list.set_intersection(pl.col("bt")).list.len().alias("k"),
                       pl.col("at").list.len().alias("na"), pl.col("bt").list.len().alias("nb"))
    ca = d.select((pl.col("k") / pl.col("na").clip(lower_bound=1)).alias("v"))["v"].to_numpy().astype(np.float32)
    cb = d.select((pl.col("k") / pl.col("nb").clip(lower_bound=1)).alias("v"))["v"].to_numpy().astype(np.float32)
    return ca, cb


def string_features(a: pl.DataFrame, b: pl.DataFrame) -> dict[str, np.ndarray]:
    """a: S1 side, b: pool side (row-aligned pairs), columns core1 name2 addr legal ctry nl_name nl_addr."""
    ac, bc = a["core1"].to_list(), b["core1"].to_list()
    aa, ba = a["addr"].to_list(), b["addr"].to_list()
    f: dict[str, np.ndarray] = {}
    for nm, sc in (("ratio", fuzz.ratio), ("partial", fuzz.partial_ratio), ("tsort", fuzz.token_sort_ratio),
                   ("tset", fuzz.token_set_ratio)):
        f[f"name_{nm}"] = _cp(ac, bc, sc)
        f[f"addr_{nm}"] = _cp(aa, ba, sc)
    f["name_jw"] = _cp(ac, bc, JaroWinkler.normalized_similarity)
    f["name_lev"] = _cp(ac, bc, Levenshtein.normalized_similarity)
    f["addr_lev"] = _cp(aa, ba, Levenshtein.normalized_similarity)
    # alias / domain label of the pool record (name2) against the S1 core name, and the reverse
    b2, a2 = b["name2"].to_list(), a["name2"].to_list()
    f["alias_tset"] = np.maximum(_cp(ac, b2, fuzz.token_set_ratio), _cp(a2, bc, fuzz.token_set_ratio))
    f["name_nospace_ratio"] = _cp([x.replace(" ", "") for x in ac], [x.replace(" ", "") for x in bc], fuzz.ratio)

    d = pl.DataFrame({"a": aa, "b": ba}).with_columns(
        pl.col("a").str.extract_all(r"\d+").alias("an"), pl.col("b").str.extract_all(r"\d+").alias("bn"))
    d = d.with_columns(
        pl.col("an").list.set_intersection(pl.col("bn")).list.len().alias("common"),
        pl.col("an").list.len().alias("na"), pl.col("bn").list.len().alias("nb"),
        pl.col("an").list.first().alias("ah"), pl.col("bn").list.first().alias("bh"),
        pl.col("an").list.eval(pl.element().filter(pl.element().str.len_chars() >= 5)).alias("apin"),
        pl.col("bn").list.eval(pl.element().filter(pl.element().str.len_chars() >= 5)).alias("bpin"),
    ).with_columns(
        pl.col("apin").list.set_intersection(pl.col("bpin")).list.len().alias("pin_common"),
        (pl.col("apin").list.len() > 0).alias("a_has_pin"), (pl.col("bpin").list.len() > 0).alias("b_has_pin"),
    )
    f["num_common"] = d["common"].to_numpy().astype(np.float32)
    f["num_common_frac"] = d.select((pl.col("common") / pl.max_horizontal("na", "nb", pl.lit(1))).alias("v"))["v"].to_numpy().astype(np.float32)
    f["house_eq"] = (d["ah"].is_not_null() & (d["ah"] == d["bh"])).to_numpy().astype(np.float32)
    f["pin_match"] = (d["pin_common"] > 0).to_numpy().astype(np.float32)
    f["pin_conflict"] = (d["a_has_pin"] & d["b_has_pin"] & (d["pin_common"] == 0)).to_numpy().astype(np.float32)
    len_b = b["addr"].str.len_chars().to_numpy()
    f["len_core_a"] = a["core1"].str.len_chars().to_numpy().astype(np.float32)
    f["len_core_b"] = b["core1"].str.len_chars().to_numpy().astype(np.float32)
    f["len_addr_b"] = len_b.astype(np.float32)
    f["addr_b_empty"] = (len_b == 0).astype(np.float32)
    f["nl_name_b"] = b["nl_name"].to_numpy().astype(np.float32)
    f["nl_addr_b"] = b["nl_addr"].to_numpy().astype(np.float32)
    f["same_ctry"] = (a["ctry"].to_numpy() == b["ctry"].to_numpy()).astype(np.float32)
    la, lb = a["legal"], b["legal"]
    f["legal_eq"] = ((la != "") & (la == lb)).to_numpy().astype(np.float32)
    f["legal_conflict"] = ((la != "") & (lb != "") & (la != lb)).to_numpy().astype(np.float32)
    # --- rarity / genericness of the name (how many S1 / pool records share it) and exact-name flag
    f["core_eq"] = ((a["core1"] != "") & (a["core1"] == b["core1"])).to_numpy().astype(np.float32)
    f["log_cnt_s1_a"] = np.log1p(a["cnt_s1_a"].to_numpy()).astype(np.float32)
    f["log_cnt_pool_b"] = np.log1p(b["cnt_pool_b"].to_numpy()).astype(np.float32)
    f["log_cnt_s1_b"] = np.log1p(b["cnt_s1_b"].to_numpy()).astype(np.float32)
    f["name_cov_a"], f["name_cov_b"] = _coverage(ac, bc)
    f["addr_cov_a"], f["addr_cov_b"] = _coverage(aa, ba)
    for side, df_ in (("a", a), ("b", b)):
        n_sp = df_["addr"].str.count_matches(" ", literal=True).to_numpy()
        f[f"ntok_addr_{side}"] = np.where(df_["addr"].str.len_chars().to_numpy() > 0, n_sp + 1, 0).astype(np.float32)
    # --- glued / handle style names: compare with spaces removed
    nsa = a["core1"].str.replace_all(" ", "", literal=True).to_list()
    nsb = b["core1"].str.replace_all(" ", "", literal=True).to_list()
    f["nospace_partial"] = _cp(nsa, nsb, fuzz.partial_ratio)
    f["nospace_jw"] = _cp(nsa, nsb, JaroWinkler.normalized_similarity)
    # --- digit alignment: all digits of the address as one string, and the house number edit distance
    da = a["addr"].str.replace_all(r"\D", "").to_list()
    db = b["addr"].str.replace_all(r"\D", "").to_list()
    f["digits_ratio"] = _cp(da, db, fuzz.ratio)
    f["digits_lev"] = _cp(da, db, Levenshtein.distance)
    ha = [x if x is not None else "" for x in d["ah"].to_list()]
    hb = [x if x is not None else "" for x in d["bh"].to_list()]
    hl = _cp(ha, hb, Levenshtein.distance)
    f["house_lev"] = np.where(np.array([bool(x) and bool(y) for x, y in zip(ha, hb)]), hl, -1.0).astype(np.float32)
    # --- romanised copies (bridge Devanagari, Telugu, Malayalam ... to Latin) and consonant skeletons
    ra, rb = a["core_rom"].to_list(), b["core_rom"].to_list()
    f["rom_tset"] = _cp(ra, rb, fuzz.token_set_ratio)
    f["rom_partial"] = _cp(ra, rb, fuzz.partial_ratio)
    f["rom_jw"] = _cp(ra, rb, JaroWinkler.normalized_similarity)
    f["skel_ratio"] = _cp(skeleton(a["core_rom"]), skeleton(b["core_rom"]), fuzz.ratio)
    ara, arb = a["addr_rom"].to_list(), b["addr_rom"].to_list()
    f["addr_rom_tset"] = _cp(ara, arb, fuzz.token_set_ratio)
    f["addr_skel_ratio"] = _cp(skeleton(a["addr_rom"]), skeleton(b["addr_rom"]), fuzz.ratio)
    return f


def pool_index(pool: pl.DataFrame, pid: np.ndarray, n2: int) -> np.ndarray:
    """Row positions in the concatenated S2+S3 table for pool ids (src * 10_000_000 + rid)."""
    return np.where(pid < 3 * PID_BASE, pid - 2 * PID_BASE, n2 + pid - 3 * PID_BASE)


def main(argv: list[str] | None = None) -> None:
    """CLI: compute and write the feature table for all candidate pairs of a split, in chunks."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--chunk", type=int, default=1_500_000)
    ap.add_argument("--rest", action="store_true", help="train: features for the S1 OUTSIDE the sample (unbiased scoring / holdout)")
    a = ap.parse_args(argv)
    P = config.paths()
    pq = P["parquet"] / a.split
    t0 = time.time()
    drop_q = None
    frac = float(config.load().get("pairs", {}).get("drop_frac", 0.0))
    if a.split == "train" and frac > 0:
        # test-like universe: the test has fewer S1 than the train (1.73M against 2.2M), hence more pool records without an owner and smaller
        # competition and name-rarity counts. Drop a random share of the S1 outside the sample and the holdout: their pool records stay in
        # the pool as distractors and every statistic below is computed without them.
        from ber.split import holdout_q

        s1_all = pl.read_parquet(pq / "source1.parquet", columns=["rid"])["rid"].to_numpy().astype(np.int64)
        smp_q = pl.read_parquet(P["sample"] / "train_s1.parquet", columns=["rid"])["rid"].to_numpy().astype(np.int64)
        pool_q = np.setdiff1d(s1_all, np.concatenate([smp_q, holdout_q().astype(np.int64)]))
        drop_q = np.sort(np.random.default_rng(77).choice(pool_q, size=min(int(frac * len(s1_all)), len(pool_q)), replace=False))
        print(f"test-like universe: dropping {len(drop_q)} of {len(s1_all)} S1", flush=True)
    cand = blocking_features(str(P["work"] / "blocks" / a.split / "cand_*.parquet"), drop_q)
    name = a.split
    if a.split == "train":  # competition stats above used the full candidate set; features only for the wanted S1
        smp = pl.read_parquet(P["sample"] / "train_s1.parquet", columns=["rid"])
        cand = cand.join(smp.rename({"rid": "q"}), on="q", how="anti" if a.rest else "semi")
        name = "train_rest" if a.rest else "train"
        lab = pl.read_parquet(pq / "labels.parquet").with_columns(
            (pl.col("src").cast(pl.Int64) * PID_BASE + pl.col("other_rid")).alias("pid"),
            pl.col("s1_rid").alias("q"), pl.lit(1, dtype=pl.Int8).alias("label")).select("q", "pid", "label")
        cand = cand.join(lab, on=["q", "pid"], how="left").with_columns(pl.col("label").fill_null(0))
    cand = cand.sort("q", "pid")
    print(f"{a.split}: {cand.height} pairs after blocking features in {time.time() - t0:.0f}s", flush=True)

    cols = ["core1", "name2", "addr", "legal", "ctry", "nl_name", "nl_addr", "core_rom", "addr_rom"]
    s1 = pl.read_parquet(pq / "source1.parquet", columns=["rid", *cols]).sort("rid")
    s2 = pl.read_parquet(pq / "source2.parquet", columns=["rid", *cols]).sort("rid")
    s3 = pl.read_parquet(pq / "source3.parquet", columns=["rid", *cols]).sort("rid")
    pool = pl.concat([s2, s3])
    # name rarity: S1 rows sharing a core name, pool rows sharing a core name, S1 rows sharing a pool record's core name
    if config.load().get("pairs", {}).get("joint_counts", False):
        # count names over the train AND test records, so that the rarity features mean the same in both splits (the test has 21% fewer S1
        # than the train, which makes every name look rarer there when counted within the split)
        po = P["parquet"] / ("test" if a.split == "train" else "train")
        s1o = pl.read_parquet(po / "source1.parquet", columns=["core1"])
        poolo = pl.concat([pl.read_parquet(po / "source2.parquet", columns=["core1"]), pl.read_parquet(po / "source3.parquet", columns=["core1"])])
        c_s1 = pl.concat([s1.select("core1"), s1o]).group_by("core1").agg(pl.len().alias("_c_s1"))
        c_pool = pl.concat([pool.select("core1"), poolo]).group_by("core1").agg(pl.len().alias("_c_pool"))
        s1 = s1.join(c_s1, on="core1", how="left", maintain_order="left").with_columns(pl.col("_c_s1").alias("cnt_s1_a")).drop("_c_s1")
        pool = (pool.join(c_pool, on="core1", how="left", maintain_order="left").with_columns(pl.col("_c_pool").alias("cnt_pool_b")).drop("_c_pool")
                    .join(c_s1.rename({"_c_s1": "cnt_s1_b"}), on="core1", how="left", maintain_order="left").with_columns(pl.col("cnt_s1_b").fill_null(0)))
    else:
        kept = s1 if drop_q is None else s1.filter(~pl.col("rid").is_in(drop_q))
        c_s1 = kept.group_by("core1").agg(pl.len().alias("_c"))
        s1 = s1.join(c_s1, on="core1", how="left", maintain_order="left").with_columns(pl.col("_c").fill_null(0).alias("cnt_s1_a")).drop("_c")
        pool = pool.with_columns(pl.len().over("core1").alias("cnt_pool_b"))
        pool = (pool.join(c_s1.rename({"_c": "cnt_s1_b"}), on="core1", how="left", maintain_order="left")
                    .with_columns(pl.col("cnt_s1_b").fill_null(0)))
    out = P["work"] / "features" / name
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("part_*.parquet"):
        old.unlink()
    q_all, pid_all = cand["q"].to_numpy(), cand["pid"].to_numpy()
    pidx = pool_index(pool, pid_all, s2.height)
    for i, s in enumerate(range(0, cand.height, a.chunk)):
        t = time.time()
        sl = slice(s, s + a.chunk)
        sa = s1[q_all[sl]]  # rid is 0..n-1 and sorted, so row index == rid
        sb = pool[pidx[sl]]
        f = string_features(sa, sb)
        part = cand[sl].with_columns([pl.Series(k, v) for k, v in f.items()])
        part.write_parquet(out / f"part_{i:04d}.parquet", compression="zstd")
        print(f"chunk {i}: {part.height} pairs, {part.width} columns in {time.time() - t:.0f}s", flush=True)
    log_stage(f"pairs_{name}", {"chunk": a.chunk}, {"pairs": float(cand.height), "seconds": time.time() - t0})


if __name__ == "__main__":
    main()
