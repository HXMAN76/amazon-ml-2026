"""TSV reading helper. All competition files are tab-separated; reading them with the default comma separator
silently yields a single column, so the separator is always explicit."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_tsv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
