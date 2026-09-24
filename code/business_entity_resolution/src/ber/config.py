"""Params and paths. Every tunable lives in configs/params.yaml; stages read only their own section."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load(path: str | Path | None = None) -> dict:
    p = Path(path or os.environ.get("BER_PARAMS", ROOT / "configs" / "params.yaml"))
    return yaml.safe_load(p.read_text())


def paths() -> dict[str, Path]:
    """DATA: raw TSV root (contains train/ and test/). WORK: everything the pipeline writes."""
    data = Path(os.environ.get("BER_DATA", "dataset"))
    work = Path(os.environ.get("BER_WORK", "work"))
    return {"data": data, "work": work, "parquet": work / "parquet", "sample": work / "sample",
            "runs": work / "runs", "stamps": work / ".stamps"}
