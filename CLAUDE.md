# CLAUDE.md

Read `context.md` first. It has the challenge window, the budget, AWS and Modal resources, what has been verified, and the day-1 checklist.

## Rules for this repo

- Never print, paste or commit secrets (AWS keys, HF or GitHub tokens). Ask the user to create or paste them in a terminal outside Claude.
- Never download datasets, images or models on the laptop, because the home network is very slow. Use Modal (`src/amlc/modal_app.py`), Kaggle, or EC2 instead.
- Every expensive job must be sharded (`--shard/--num-shards` or `--start/--end`) and resumable. Write part files and merge them with `amlc.inference.shard`.
- Validate every submission: `python -m amlc.submission.validator <csv> --test <test>`.
- Stop compute when you are done:
  - AWS: `AWS_PROFILE=amlc bash aws/99_stop_all.sh`
  - Modal: `uv run modal app list` and `uv run modal app stop <id>`
- Use us-east-1 for everything. The hub bucket is `s3://amlc-2026-hub-567503593043`.

## Commands

- `uv sync --all-extras`: install everything
- `uv run pytest -q`: run the tests (must stay green)
- `aws login --profile amlc`: log in to AWS on the laptop. Sessions expire, so re-run it when needed.
