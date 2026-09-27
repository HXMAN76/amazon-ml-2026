"""Sanity report on the prepared Parquet: null/empty rates and side-by-side raw vs normalised samples."""

import os

import polars as pl

W = os.environ.get("BER_WORK", "work")
pq = f"{W}/parquet/"
pl.Config.set_tbl_cols(12)
pl.Config.set_fmt_str_lengths(70)
pl.Config.set_tbl_width_chars(260)
pl.Config.set_tbl_rows(14)
for sp in ("train", "test"):
    for i in (1, 2, 3):
        d = pl.read_parquet(f"{pq}{sp}/source{i}.parquet")
        c = d["ctry"].value_counts().sort("count", descending=True).head(4).to_dicts()
        print(f"== {sp} S{i} rows={d.height} empty_core1={(d['core1'] == '').mean():.5f} empty_name1={(d['name1'] == '').mean():.5f} "
              f"alias={d['has_alias'].mean():.4f} domain={d['is_domain'].mean():.4f} legal={(d['legal'] != '').mean():.3f} "
              f"empty_addr={(d['addr'] == '').mean():.4f} ctry={c}")
d = pl.read_parquet(pq + "train/source2.parquet")
print("--- non-Latin names"); print(d.filter(pl.col("nl_name") > 0.5).sample(6, seed=1).select("business_name", "name1", "business_address", "addr"))
print("--- alias/domain"); print(d.filter(pl.col("has_alias") | pl.col("is_domain")).sample(8, seed=1).select("business_name", "name1", "name2", "is_domain"))
print("--- leet candidates"); print(d.filter(pl.col("business_name").str.contains(r"[A-Za-z][0-9][A-Za-z]")).sample(8, seed=1).select("business_name", "name1"))
print("--- html"); print(d.filter(pl.col("business_name").str.contains("&amp;|&#")).head(4).select("business_name", "name1"))
print("--- US addr"); print(d.filter(pl.col("ctry") == "us").sample(6, seed=2).select("business_address", "addr"))
t = pl.read_parquet(pq + "test/source1.parquet")
print("--- France S1"); print(t.filter(pl.col("ctry") == "france").sample(6, seed=3).select("business_name", "name1", "legal", "business_address", "addr"))
