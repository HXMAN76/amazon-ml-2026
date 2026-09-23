"""Submission validator. Run before EVERY upload: a malformed file wastes a submission slot.

Fill configs/submission.yaml from the problem statement on day 1, then:
    python -m amlc.submission.validator outputs/submission.csv --test data/raw/test.csv
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl
import yaml


@dataclass
class SubmissionSpec:
    id_col: str = "sample_id"
    pred_cols: list[str] = field(default_factory=lambda: ["prediction"])
    # "float" | "int" | "str"
    pred_type: str = "float"
    min_value: float | None = None
    max_value: float | None = None
    allowed_values: list[str] | None = None
    regex: str | None = None  # e.g. 2024 format: r"^$|^\d+(\.\d+)? (gram|kilogram|...)$"
    allow_empty: bool = False
    extra_cols_ok: bool = False
    header: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SubmissionSpec":
        return cls(**(yaml.safe_load(Path(path).read_text()) or {}))


def validate(sub: pl.DataFrame, spec: SubmissionSpec, test_ids: list | None = None) -> list[str]:
    """Returns a list of problems; empty list means OK."""
    errs: list[str] = []
    expected = [spec.id_col, *spec.pred_cols]
    missing = [c for c in expected if c not in sub.columns]
    if missing:
        return [f"missing columns {missing}; have {sub.columns}"]
    extra = [c for c in sub.columns if c not in expected]
    if extra and not spec.extra_cols_ok:
        errs.append(f"unexpected extra columns {extra}")
    if sub.columns[: len(expected)] != expected:
        errs.append(f"column order {sub.columns} != expected {expected}")

    ids = sub[spec.id_col]
    if ids.null_count():
        errs.append(f"{ids.null_count()} null ids")
    if ids.is_duplicated().any():
        errs.append(f"{ids.is_duplicated().sum()} duplicated id rows, e.g. {ids.filter(ids.is_duplicated()).head(3).to_list()}")

    if test_ids is not None:
        want = {str(x) for x in test_ids}
        have = {str(x) for x in ids.to_list()}
        if len(sub) != len(test_ids):
            errs.append(f"row count {len(sub)} != test rows {len(test_ids)}")
        if want - have:
            errs.append(f"{len(want - have)} test ids missing, e.g. {sorted(want - have)[:5]}")
        if have - want:
            errs.append(f"{len(have - want)} ids not in test, e.g. {sorted(have - want)[:5]}")

    for c in spec.pred_cols:
        s = sub[c]
        if spec.pred_type in ("float", "int"):
            num = s.cast(pl.Float64, strict=False)
            bad = num.is_null().sum() - (s.is_null().sum() if spec.allow_empty else 0)
            if bad:
                errs.append(f"{c}: {bad} null / non-numeric values")
            if num.is_nan().any() or num.is_infinite().any():
                errs.append(f"{c}: NaN/inf present")
            if spec.pred_type == "int" and ((num.drop_nulls() % 1) != 0).any():
                errs.append(f"{c}: non-integer values")
            if spec.min_value is not None and (num < spec.min_value).any():
                errs.append(f"{c}: {(num < spec.min_value).sum()} values < {spec.min_value}")
            if spec.max_value is not None and (num > spec.max_value).any():
                errs.append(f"{c}: {(num > spec.max_value).sum()} values > {spec.max_value}")
        else:
            txt = s.cast(pl.String).fill_null("")
            if not spec.allow_empty and (txt == "").any():
                errs.append(f"{c}: {(txt == '').sum()} empty predictions")
            if spec.allowed_values is not None:
                bad = ~txt.is_in(spec.allowed_values + ([""] if spec.allow_empty else []))
                if bad.any():
                    errs.append(f"{c}: {bad.sum()} values outside allowed set, e.g. {txt.filter(bad).head(3).to_list()}")
            if spec.regex:
                rx = re.compile(spec.regex)
                bad_vals = [v for v in txt.to_list() if not rx.fullmatch(v)]
                if bad_vals:
                    errs.append(f"{c}: {len(bad_vals)} values fail format regex, e.g. {bad_vals[:3]}")
    return errs


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("submission")
    p.add_argument("--spec", default="configs/submission.yaml")
    p.add_argument("--test", help="test csv/parquet to check id coverage")
    a = p.parse_args(argv)

    spec = SubmissionSpec.from_yaml(a.spec)
    sub = pl.read_csv(a.submission, infer_schema_length=0, has_header=spec.header)
    test_ids = None
    if a.test:
        from amlc.data.io import read_table

        test_ids = read_table(a.test)[spec.id_col].to_list()
    errs = validate(sub, spec, test_ids)
    if errs:
        print("INVALID SUBMISSION:")
        for e in errs:
            print("  -", e)
        sys.exit(1)
    print(f"OK: {len(sub)} rows, columns {sub.columns}")


if __name__ == "__main__":
    main()
