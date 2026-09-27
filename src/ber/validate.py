"""Local re-implementation of the format rules (the official utils/validate_submission.py is authoritative)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ber.data import read_tsv


def _raw_tsv(path: str | Path) -> pd.DataFrame:
    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        rows = [line.rstrip("\n").partition("\t") for line in f if line.strip()]
    return pd.DataFrame({header[0]: [r[0] for r in rows], header[1]: [r[2] for r in rows]})


def validate(matching: str | Path, candidate: str | Path, test_dir: str | Path) -> list[str]:
    """Check both output files against the format rules (raw-line parsing, like the official validator)."""
    test_dir = Path(test_dir)
    s1 = set(read_tsv(test_dir / "test_source1.tsv")["entity_id"])
    others = set(read_tsv(test_dir / "test_source2.tsv")["entity_id"]) | set(read_tsv(test_dir / "test_source3.tsv")["entity_id"])
    issues: list[str] = []
    lists: dict[str, dict[str, set[str]]] = {}
    for name, path, col in (("matching", matching, "matched_entity_ids"), ("candidate", candidate, "candidate_entity_ids")):
        df = _raw_tsv(path)  # raw lines like the official validator: no CSV un-quoting (it hid a `""` bug once)
        if list(df.columns) != ["source1_entity_id", col]:
            issues.append(f"{name}: bad columns {list(df.columns)}")
            continue
        if df["source1_entity_id"].duplicated().any():
            issues.append(f"{name}: duplicate source1_entity_id rows")
        if set(df["source1_entity_id"]) != s1:
            issues.append(f"{name}: source1 ids differ from test_source1 ({len(set(df['source1_entity_id']) ^ s1)} mismatches)")
        d: dict[str, set[str]] = {}
        for a, b in zip(df["source1_entity_id"], df[col]):
            ids = [x for x in b.split(",") if x]
            if len(ids) != len(set(ids)):
                issues.append(f"{name}: duplicate ids in list for {a}")
            bad = [x for x in ids if x not in others]
            if bad:
                issues.append(f"{name}: unknown/non-S2S3 ids for {a}: {bad[:3]}")
            d[a] = set(ids)
        lists[name] = d
    if len(lists) == 2:
        extra = sum(len(v - lists["candidate"].get(k, set())) for k, v in lists["matching"].items())
        if extra:
            issues.append(f"warning: {extra} matched ids not in candidate set")
    return issues
