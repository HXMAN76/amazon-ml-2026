"""Data loading, image index and S3 helpers."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import polars as pl

from amlc.data.downloader import MANIFEST

# Shared data hub bucket in Account A. Override with env var if you name it differently.
BUCKET = os.environ.get("AMLC_BUCKET", "")


def read_table(path: str | Path) -> pl.DataFrame:
    path = str(path)
    if path.endswith(".parquet"):
        return pl.read_parquet(path)
    # Past datasets had multi-line catalog text in quoted fields; polars handles it, pandas' C engine
    # sometimes does not. Pass schema_overrides later if an id column must stay a string.
    return pl.read_csv(path, infer_schema_length=10000)


def image_index(image_dir: str | Path) -> pl.DataFrame:
    """url -> local file path for every successfully downloaded image (from the manifest)."""
    image_dir = Path(image_dir)
    rows = []
    with (image_dir / MANIFEST).open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("status") == "ok":
                rows.append((r["url"], str(image_dir / r["file"])))
    return pl.DataFrame(rows, schema=["url", "image_path"], orient="row").unique("url", keep="last")


def attach_images(df: pl.DataFrame, image_dir: str | Path, url_col: str = "image_link") -> pl.DataFrame:
    """Left-join local image paths onto a table; missing downloads get null image_path."""
    idx = image_index(image_dir).rename({"url": url_col})
    return df.join(idx, on=url_col, how="left")


def s3_uri(key: str) -> str:
    if not BUCKET:
        raise RuntimeError("set AMLC_BUCKET env var to the data hub bucket name")
    return f"s3://{BUCKET}/{key.lstrip('/')}"


def s3_sync(src: str, dst: str, *extra: str) -> None:
    """`aws s3 sync` is far faster than per-object boto3 for 100k+ small files."""
    cmd = ["aws", "s3", "sync", src, dst, "--only-show-errors", *extra]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main(argv: list[str] | None = None) -> None:
    """Attach local image paths to a table: the input every image job (embed / vlm) expects.

    python -m amlc.data.io --input data/raw/test.csv --images data/images --out data/test_img.parquet
    """
    import argparse

    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--images", required=True, help="downloader output dir (has manifest.jsonl)")
    p.add_argument("--url-col", default="image_link")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    df = attach_images(read_table(a.input), a.images, a.url_col)
    df.write_parquet(a.out)
    miss = df["image_path"].null_count()
    print(f"{df.height} rows -> {a.out}; {miss} without image ({miss / max(df.height, 1):.2%})")


if __name__ == "__main__":
    main()
