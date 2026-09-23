"""Row-range sharding so any job can be split across Kaggle / Colab / EC2 / laptop workers.

Every sharded job:
    * takes --start/--end (row slice of the input table) or --shard i --num-shards n
    * writes <out_dir>/part-<start>-<end>.parquet (row ids included)
    * skips the shard if that file already exists (safe to rerun after a crash)
Then `merge_parts(out_dir)` concatenates and checks for gaps/overlaps.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl


def shard_bounds(n_rows: int, shard: int, num_shards: int) -> tuple[int, int]:
    if not 0 <= shard < num_shards:
        raise ValueError(f"shard {shard} not in [0, {num_shards})")
    size = -(-n_rows // num_shards)  # ceil
    start = min(shard * size, n_rows)
    return start, min(start + size, n_rows)


def add_shard_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--start", type=int, default=None)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--shard", type=int, default=None, help="shard index (alternative to --start/--end)")
    p.add_argument("--num-shards", type=int, default=None)


def resolve_range(args: argparse.Namespace, n_rows: int) -> tuple[int, int]:
    if args.shard is not None:
        if args.num_shards is None:
            raise SystemExit("--shard needs --num-shards")
        return shard_bounds(n_rows, args.shard, args.num_shards)
    start = args.start or 0
    end = n_rows if args.end is None else min(args.end, n_rows)
    return start, end


def part_path(out_dir: str | Path, start: int, end: int) -> Path:
    return Path(out_dir) / f"part-{start:09d}-{end:09d}.parquet"


def merge_parts(out_dir: str | Path, expected_rows: int | None = None, row_col: str = "row") -> pl.DataFrame:
    parts = sorted(Path(out_dir).glob("part-*.parquet"))
    if not parts:
        raise FileNotFoundError(f"no part-*.parquet in {out_dir}")
    df = pl.concat([pl.read_parquet(p) for p in parts], how="vertical_relaxed").sort(row_col)
    dup = df[row_col].is_duplicated().sum()
    if dup:
        raise ValueError(f"{dup} duplicated rows across parts (overlapping shards?)")
    if expected_rows is not None and df.height != expected_rows:
        have = set(df[row_col].to_list())
        missing = [i for i in range(expected_rows) if i not in have]
        raise ValueError(f"expected {expected_rows} rows, got {df.height}; first missing rows: {missing[:10]}")
    return df


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="merge shard outputs")
    p.add_argument("out_dir")
    p.add_argument("--expected-rows", type=int, default=None)
    p.add_argument("--to", required=True, help="merged parquet path")
    a = p.parse_args(argv)
    df = merge_parts(a.out_dir, a.expected_rows)
    df.write_parquet(a.to)
    print(f"merged {df.height} rows -> {a.to}")


if __name__ == "__main__":
    main()
