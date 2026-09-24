"""Loading TSVs, building normalised record tables and the ground-truth pair set."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ber import normalize as N


def read_tsv(path: str | Path) -> pd.DataFrame:
    # explicit sep: reading a .tsv with the default comma separator silently yields one column
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["name"] = df["business_name"].map(N.norm_name)
    df["core"] = df["business_name"].map(N.core_name)
    df["addr"] = df["business_address"].map(N.norm_addr)
    df["ctry"] = df["country"].map(N.norm_country)
    df["nums"] = df["addr"].map(N.numbers)
    df["both"] = df["core"] + " " + df["addr"]
    return df.reset_index(drop=True)


@dataclass
class Split:
    s1: pd.DataFrame
    s2: pd.DataFrame
    s3: pd.DataFrame
    truth: dict[str, set[str]] | None  # s1 id -> set of matching S2/S3 ids

    @property
    def others(self) -> pd.DataFrame:
        return pd.concat([self.s2, self.s3], ignore_index=True)


def load_truth(path: str | Path) -> dict[str, set[str]]:
    gt = read_tsv(path)
    out: dict[str, set[str]] = {}
    for a, b in zip(gt["source1_entity_id"], gt["matched_entity_ids"]):
        out[a] = {x.strip() for x in b.split(",") if x.strip()}
    return out


def load_split(root: str | Path, split: str) -> Split:
    root = Path(root) / split
    s1, s2, s3 = (prep(read_tsv(root / f"{split}_source{i}.tsv")) for i in (1, 2, 3))
    gt_path = root / f"{split}_ground_truth.tsv"
    truth = load_truth(gt_path) if gt_path.exists() else None
    if truth is not None:
        for e in s1["entity_id"]:
            truth.setdefault(e, set())
    return Split(s1, s2, s3, truth)
