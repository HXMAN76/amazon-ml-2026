"""Language-free sibling / noise features added to existing stack chunks. Usage: python src/scripts/france/stack_langfree.py TAG_IN TAG_OUT
Copies WORK/stack{TAG_IN}/{train,test}/chunk_*.parquet to WORK/stack{TAG_OUT}/ with extra columns computed from the split's own records, per country:
  lf_n_miss, lf_n_extra, lf_n_common  words only in the S1 core, only in the pool core, shared
  lf_miss_maxdf, lf_extra_mindf       log document frequency (S1 cores of the country) of the differing words: two common words swapped is a sibling
  lf_extra_maxratio, lf_miss_minratio log of (share of pool cores holding the word) / (share of S1 cores holding it): words the generator injects
                                      into copies sit far above 0 (noise), type words near 0
  lf_legal_disj                       both records carry a legal form (joined or spaced, `s a r l`) and they share none
  lf_log_ns                           log count of the country's S1 with the same core (namesakes)
  lf_coined                           the pool core is one 6+ letter word absent from the country's S1 vocabulary
The model learns from US/India labels what these mean and applies it to France's own words, which it has never seen."""

import sys

import polars as pl

from ber import config
from france_variants import legal_set

PID_BASE = 10_000_000


def tables(split: str):
    pq = config.paths()["parquet"] / split
    s1 = pl.read_parquet(pq / "source1.parquet", columns=["rid", "core1", "name1", "legal", "ctry"]).select(
        pl.col("rid").cast(pl.Int64).alias("q"), pl.col("core1").alias("a_core"), legal_set(pl.col("name1"), pl.col("legal")).alias("la"), "ctry")
    pool = pl.concat([pl.read_parquet(pq / f"source{s}.parquet", columns=["rid", "core1", "name1", "legal", "ctry"]).select(
        (pl.col("rid").cast(pl.Int64) + s * PID_BASE).alias("pid"), pl.col("core1").alias("b_core"), legal_set(pl.col("name1"), pl.col("legal")).alias("lb"),
        pl.col("ctry").alias("b_ctry")) for s in (2, 3)])
    words = lambda d, col, c: d.select(c, pl.col(col).str.split(" ").list.unique().alias("w")).explode("w").group_by(c, "w").len()
    ws = words(s1, "a_core", "ctry").rename({"len": "n_s1"})
    wp = words(pool, "b_core", "b_ctry").rename({"b_ctry": "ctry", "len": "n_pool"})
    n1 = s1.group_by("ctry").len().rename({"len": "tot_s1"})
    n2 = pool.group_by("b_ctry").len().rename({"b_ctry": "ctry", "len": "tot_pool"})
    st = (ws.join(wp, on=["ctry", "w"], how="full", coalesce=True).fill_null(0).join(n1, on="ctry").join(n2, on="ctry")
            .select("ctry", "w", (pl.col("n_s1") + 1).log().alias("ldf"),
                    (((pl.col("n_pool") + 1) / pl.col("tot_pool")).log() - ((pl.col("n_s1") + 1) / pl.col("tot_s1")).log()).alias("lratio"),
                    (pl.col("n_s1") > 0).alias("in_vocab")))
    s1 = s1.join(s1.group_by("ctry", "a_core").len().rename({"len": "n_ns"}), on=["ctry", "a_core"], how="left")
    return s1, pool, st


def features(d: pl.DataFrame, s1: pl.DataFrame, pool: pl.DataFrame, st: pl.DataFrame) -> pl.DataFrame:
    x = d.select("q", "pid").join(s1, on="q", how="left").join(pool, on="pid", how="left")
    ta, tb = pl.col("a_core").str.split(" ").list.unique(), pl.col("b_core").str.split(" ").list.unique()
    x = x.with_columns(ta.list.set_difference(tb).alias("_miss"), tb.list.set_difference(ta).alias("_extra"), ta.list.set_intersection(tb).list.len().alias("lf_n_common"))
    agg = []
    for col, f in (("_miss", [pl.col("ldf").max().alias("lf_miss_maxdf"), pl.col("lratio").min().alias("lf_miss_minratio")]),
                   ("_extra", [pl.col("ldf").min().alias("lf_extra_mindf"), pl.col("lratio").max().alias("lf_extra_maxratio")])):
        e = x.select("q", "pid", "ctry", pl.col(col).alias("w")).explode("w").join(st, on=["ctry", "w"], how="left")
        agg.append(e.group_by("q", "pid").agg(*f))
    x = x.join(agg[0], on=["q", "pid"], how="left").join(agg[1], on=["q", "pid"], how="left")
    single = st.filter(pl.col("in_vocab")).select("ctry", pl.col("w").alias("b_core"), pl.lit(True).alias("_voc"))
    x = x.join(single, on=["ctry", "b_core"], how="left")
    return x.select("q", "pid",
                    pl.col("_miss").list.len().cast(pl.Float32).alias("lf_n_miss"), pl.col("_extra").list.len().cast(pl.Float32).alias("lf_n_extra"),
                    pl.col("lf_n_common").cast(pl.Float32), pl.col("lf_miss_maxdf").cast(pl.Float32), pl.col("lf_miss_minratio").cast(pl.Float32),
                    pl.col("lf_extra_mindf").cast(pl.Float32), pl.col("lf_extra_maxratio").cast(pl.Float32),
                    ((pl.col("la").list.len() > 0) & (pl.col("lb").list.len() > 0) & (pl.col("la").list.set_intersection(pl.col("lb")).list.len() == 0)).cast(pl.Float32).alias("lf_legal_disj"),
                    (pl.col("n_ns").fill_null(1) + 0.0).log().cast(pl.Float32).alias("lf_log_ns"),
                    (pl.col("b_core").str.contains(r"^[a-z]{6,}$") & pl.col("_voc").is_null()).cast(pl.Float32).alias("lf_coined"))


def main() -> None:
    tin, tout = sys.argv[1], sys.argv[2]
    W = config.paths()["work"]
    for split in ("train", "test"):
        s1, pool, st = tables(split)
        (W / f"stack{tout}" / split).mkdir(parents=True, exist_ok=True)
        for f in sorted((W / f"stack{tin}" / split).glob("chunk_*.parquet")):
            d = pl.read_parquet(f).with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
            d = d.join(features(d, s1, pool, st), on=["q", "pid"], how="left")
            d.write_parquet(W / f"stack{tout}" / split / f.name, compression="zstd")
            print(f"{split} {f.name}: {d.height} pairs, {d.width} columns; lf_legal_disj {d['lf_legal_disj'].mean():.4f}, lf_coined {d['lf_coined'].mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
