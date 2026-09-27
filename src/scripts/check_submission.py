"""Check a pair of output files against every rule of the challenge statement, and report per-country statistics.

Usage: python src/scripts/check_submission.py OUT_DIR TEST_DIR
  OUT_DIR  folder with matching_results.tsv and candidate_pairs.tsv
  TEST_DIR folder with test_source1.tsv, test_source2.tsv, test_source3.tsv

Rules checked (from the problem statement):
  file format (UTF-8, LF, tab separated, exact header, no quoting), one row per test S1 entity and no duplicates,
  empty list for singletons, no duplicate ids in a list, only S2-/S3- ids that exist in the test set (no S1 self-matches),
  every matched id appears in candidate_pairs.tsv, candidate file has the same S1 rows.
Extra sanity information: one owner per S2/S3 record, match rates per country (France is absent from training).
"""

import collections
import sys
from pathlib import Path

MATCH_HEADER = "source1_entity_id\tmatched_entity_ids"
CAND_HEADER = "source1_entity_id\tcandidate_entity_ids"


def _read_ids(path: Path, with_country: bool = False):
    """Entity ids (and countries) of a test source file; parsed by tab, no CSV quoting."""
    ids, ctry = [], {}
    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        ci = header.index("country") if "country" in header else None
        for line in f:
            parts = line.rstrip("\n").split("\t")
            ids.append(parts[0])
            if with_country and ci is not None and len(parts) > ci:
                ctry[parts[0]] = parts[ci]
    return (ids, ctry) if with_country else ids


def _raw_rows(path: Path, header: str, issues: list[str], name: str):
    """Yield (s1_id, list_text) from a results-style file while checking byte-level format rules."""
    raw = path.read_bytes()[:3]
    if raw.startswith(b"\xef\xbb\xbf"):
        issues.append(f"{name}: has a UTF-8 BOM")
    with open(path, encoding="utf-8", newline="") as f:  # newline="" keeps any CR visible
        first = f.readline()
        if first.rstrip("\n") != header:
            issues.append(f"{name}: header is {first.rstrip(chr(10))!r}, expected {header!r}")
        for n, line in enumerate(f, start=2):
            if "\r" in line:
                issues.append(f"{name}: carriage return at line {n}")
                break
            body = line.rstrip("\n")
            if body.count("\t") != 1:
                issues.append(f"{name}: line {n} has {body.count(chr(9))} tabs")
                break
            if '"' in body:
                issues.append(f"{name}: line {n} contains a quote character")
                break
            s1, _, rest = body.partition("\t")
            yield s1, rest


def check(out_dir: str | Path, test_dir: str | Path, verbose: bool = True) -> dict:
    """Run all checks; returns {"issues": [...], "stats": {...}}. An empty issue list means the files satisfy the rules."""
    out_dir, test_dir = Path(out_dir), Path(test_dir)
    issues: list[str] = []
    s1_ids, ctry = _read_ids(test_dir / "test_source1.tsv", with_country=True)
    s1_set = set(s1_ids)
    pool = set(_read_ids(test_dir / "test_source2.tsv")) | set(_read_ids(test_dir / "test_source3.tsv"))

    cand_line: dict[str, str] = {}
    for s1, rest in _raw_rows(out_dir / "candidate_pairs.tsv", CAND_HEADER, issues, "candidate_pairs.tsv"):
        if s1 in cand_line:
            issues.append(f"candidate_pairs.tsv: duplicate S1 row {s1}")
        cand_line[s1] = rest

    seen: set[str] = set()
    owner: dict[str, str] = {}
    by_ctry = collections.defaultdict(lambda: [0, 0, 0, 0])  # rows, non-empty, matched ids, candidate ids
    src_share = collections.Counter()
    subset_violations = unknown = self_match = dup_in_list = shared_records = 0
    for s1, rest in _raw_rows(out_dir / "matching_results.tsv", MATCH_HEADER, issues, "matching_results.tsv"):
        if s1 in seen:
            issues.append(f"matching_results.tsv: duplicate S1 row {s1}")
        seen.add(s1)
        ids = [x for x in rest.split(",")] if rest else []
        if len(ids) != len(set(ids)):
            dup_in_list += 1
        cands = set(cand_line.get(s1, "").split(",")) if s1 in cand_line else set()
        for i in ids:
            if i.startswith("S1-"):
                self_match += 1
            elif not i.startswith(("S2-", "S3-")) or i not in pool:
                unknown += 1
            if i not in cands:
                subset_violations += 1
            if i in owner and owner[i] != s1:
                shared_records += 1
            owner[i] = s1
            src_share[i[:2]] += 1
        c = by_ctry[ctry.get(s1, "?")]
        c[0] += 1
        c[1] += bool(ids)
        c[2] += len(ids)
        c[3] += len(cand_line.get(s1, "").split(",")) if cand_line.get(s1) else 0

    if seen != s1_set:
        issues.append(f"matching_results.tsv: S1 ids differ from test_source1 ({len(seen ^ s1_set)} mismatches)")
    if set(cand_line) != s1_set:
        issues.append(f"candidate_pairs.tsv: S1 ids differ from test_source1 ({len(set(cand_line) ^ s1_set)} mismatches)")
    for name, n in (("lists with duplicate ids", dup_in_list), ("S1 self-matches", self_match),
                    ("ids that are not S2-/S3- ids of the test set", unknown),
                    ("matched ids missing from candidate_pairs.tsv", subset_violations),
                    ("S2/S3 records matched to more than one S1", shared_records)):
        if n:
            issues.append(f"matching_results.tsv: {n} {name}")

    stats = {"rows": len(seen), "countries": {
        k: {"rows": v[0], "matched_share": v[1] / max(v[0], 1), "mean_matches": v[2] / max(v[0], 1),
            "mean_candidates": v[3] / max(v[0], 1)} for k, v in sorted(by_ctry.items())},
        "source_share_of_matches": {k: v / max(sum(src_share.values()), 1) for k, v in sorted(src_share.items())}}
    if verbose:
        print("ISSUES:", "none (all rules satisfied)" if not issues else issues)
        print(f"rows {stats['rows']} (test S1 {len(s1_set)})")
        for k, v in stats["countries"].items():
            print(f"  {k:8s} rows {v['rows']:8d} matched_share {v['matched_share']:.3f} "
                  f"mean_matches {v['mean_matches']:.2f} mean_candidates {v['mean_candidates']:.1f}")
        print("share of matched ids by source:", {k: round(v, 3) for k, v in stats["source_share_of_matches"].items()})
    return {"issues": issues, "stats": stats}


if __name__ == "__main__":
    res = check(sys.argv[1], sys.argv[2])
    sys.exit(1 if res["issues"] else 0)
