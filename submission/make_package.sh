#!/bin/bash
# Build the final submission zip in the organisers' layout (docs/README.md "Final Submission Package"):
#   Nooglers_submission.zip: output/{matching_results.tsv,candidate_pairs.tsv}, code/business_entity_resolution/{src,README.md,requirements.txt},
#   Documentation_template.md
# Usage (from the repository root): bash submission/make_package.sh <RUN_NAME>   e.g. v8u_s27_AR
# The two TSV files are taken from the team folder runs/<RUN_NAME>/output/ (validated there with --check-ids).
set -euo pipefail
RUN=${1:?usage: make_package.sh <RUN_NAME>}
S3=s3://ml-challenge-nooglers/ml-challenge-2026/handoff-nooglers-20260926/runs/$RUN/output
OUT=dist/Nooglers_submission
rm -rf "$OUT" "$OUT.zip"
mkdir -p "$OUT/output" "$OUT/code/business_entity_resolution"
aws s3 cp "$S3/matching_results.tsv" "$OUT/output/" --only-show-errors
aws s3 cp "$S3/candidate_pairs.tsv" "$OUT/output/" --only-show-errors
PKG=code/business_entity_resolution   # layout inside the zip; the sources live at the repository root
mkdir -p "$OUT/$PKG/src"
rsync -a --exclude '__pycache__' --exclude '*.pyc' --exclude '.pytest_cache' src/ber src/scripts src/tests "$OUT/$PKG/src/"
cp docs/pipeline.md "$OUT/$PKG/README.md"
cp requirements.txt requirements-gpu.txt reproduce_final.sh "$OUT/$PKG/"
cp -r configs "$OUT/$PKG/"
cp submission/Documentation_template.md "$OUT/"
(cd dist && zip -qr Nooglers_submission.zip Nooglers_submission)
python3 - "$OUT/output" <<'PY'
import sys, pathlib
d = pathlib.Path(sys.argv[1])
for f in ("matching_results.tsv", "candidate_pairs.tsv"):
    lines = (d / f).read_text().splitlines()
    print(f, len(lines) - 1, "S1 rows; header:", lines[0])
PY
ls -la dist/Nooglers_submission.zip
