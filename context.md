# Project context: Amazon ML Challenge 2026

Handoff notes for any new session (human or Claude). Last updated 2026-09-24.

## Challenge

- Hosted on Unstop. Team of 4 college students, led by HXMAN76.
- The assessment window runs from **24 Sep 2026 18:30 UTC to 27 Sep 2026 18:29 UTC**, which is 25 Sep 00:00 IST to 27 Sep 23:59 IST (72 hours).
- Ranking uses the best score. **Ties go to whoever submitted earlier**, so submit a valid baseline early.
- Required deliverables: a 1–2 page approach document and a zip of the code. The finale is on 7 Oct 2026, for the top 10 teams.
- The task is unknown until release. Past years:
  - 2024: extract entity values from product images; metric F1; about 263k train and 131k test rows.
  - 2025: predict price from multimodal catalog text plus images; metric SMAPE. There were model-size and license restrictions.
- **Always check this year's rules** for model limits, external data, the metric and the submission limit.

## Compute and budget

| Pool | Status | Notes |
|---|---|---|
| Laptop | RTX 4060 8GB, 24 cores, 30GB RAM, Fedora | The home network is very slow (~50 KB/s–1.5 MB/s). **Never download datasets, images or models on the laptop.** |
| Kaggle ×4 | Each member sets up their own | T4×2 or P100, 30 GPU-h/week each; the quota resets Saturday 00:00 UTC (inside the window). Bootstrap: `notebooks/kaggle_bootstrap.py` |
| Modal | Account A's workspace is verified working | $30/month credit per workspace. **A card is required.** Set spend limit $0 so only credits are used. Up to 10 GPUs in parallel |
| AWS account A (567503593043) | Paid plan, $200 credits | Budget alert at $180 to the owner's email. GPU quota 4 vCPU on-demand plus 4 spot, case open (not approved as of 24 Sep) |
| AWS accounts B/C/D | Free plan, $200 each | Free plan cannot launch GPUs. Use them for S3 and CPU only, or keep them in reserve |

- Region is **us-east-1** for everything.
- Hub bucket: `s3://amlc-2026-hub-567503593043`.
  - Prefixes 00-raw through 07-experiments.
  - 00-raw cannot be deleted.
  - `tmp/` expires after 3 days.
- IAM:
  - `amlc-compute-role` is the EC2/SageMaker instance profile, with hub access.
  - `amlc-external` is an S3-only user for Kaggle and Modal. Its key was rotated on 2026-09-23 after the first key leaked into a chat transcript.
- Laptop AWS access: `aws login --profile amlc`. It is a short-lived root session, so re-run it when it expires.
- **Do not join AWS Organizations.** That forces the Paid plan on every account.

GPU prices, checked via the AWS Pricing API on 2026-09-23:

| Instance | us-east-1 on-demand | Mumbai spot |
|---|---|---|
| g4dn.xlarge | $0.53/h | $0.19/h |
| g6.xlarge | $0.80/h | — |
| g5.xlarge | $1.01/h | — |

## Repo layout (`~/amazon-ml-2026`, private GitHub `HXMAN76/amazon-ml-2026`)

- `src/amlc/data/downloader.py`: async, resumable image downloader. It retries with backoff, writes `manifest.jsonl` and `failed.csv`, and can resize.
- `src/amlc/data/io.py`: table IO. `python -m amlc.data.io` attaches `image_path` to a table.
- `src/amlc/features/embed.py`: sharded image and text embeddings (SigLIP/CLIP/sentence-transformers).
- `src/amlc/features/text.py`: TF-IDF+SVD, numeric stats, and value/unit parsing.
- `src/amlc/inference/vlm.py`: sharded vision-language-model inference. Optional 4/8-bit, checkpointed every 500 rows.
- `src/amlc/inference/shard.py`: `--start/--end` or `--shard/--num-shards`, then merge with gap and overlap checks.
- `src/amlc/models/gbm.py`: k-fold LightGBM/XGBoost/CatBoost with OOF predictions, plus `blend()`.
- `src/amlc/training/finetune_text.py`: LoRA/QLoRA fine-tuning that resumes from the latest checkpoint.
- `src/amlc/baseline.py`: one command from TF-IDF to a validated submission.
- `src/amlc/submission/validator.py` and `configs/submission.yaml`: fill these on day 1.
- `src/amlc/evaluation.py`: metrics smape, rmse, mae, accuracy, f1_macro and extraction_f1.
- `src/amlc/tracking.py`: MLflow (sqlite) plus an append-only `artifacts/runs.jsonl`.
- `src/amlc/modal_app.py`: Modal fan-out with presets `embed-image`, `embed-text`, `vlm`, or a custom `--script`.
- `aws/*.sh`: CloudShell and laptop scripts.
  - 00: account check
  - 01: quotas
  - 02: budget
  - 10: hub bucket
  - 15: external user
  - 20: compute role
  - 30: verify access
  - 40: launch GPU (auto-terminates)
  - 99: stop everything
- Tests: `uv run pytest` (11 tests, all passing).

## Verified

- Local tests pass. Torch 2.14 with cu130 sees the 4060.
- On Modal T4, all three presets were verified end to end on a 24-row smoke set in `s3://.../tmp/smoke/`: text embeddings, image download plus SigLIP2, and SmolVLM.

## Known gaps and decisions

- Plain HF `generate` for the vision-language model runs at about 0.5 rows/s per T4. If the task needs one over about 100k images, add a vLLM backend (roughly 5–20× faster).
- These were chosen instead of Terraform and SageMaker: plain CLI scripts, and EC2 or Modal for compute.
- The library stack is transformers v5, which returns ModelOutput objects from `get_*_features`. `_pooled()` in embed.py handles that.
- Secrets never go in the chat or in git. Create keys in a terminal outside Claude.

## Day-1 checklist

1. Read the rules. Fill `configs/submission.yaml` and add the metric to `evaluation.py` if it is new.
2. Upload the raw data once to `00-raw/`. Download images on Modal or Kaggle, never on the laptop.
3. Within about 2 hours, submit a baseline with `python -m amlc.baseline ...` and validate it first.
4. Split the work into parallel tracks:
   - A: data and submissions
   - B: text features and GBMs
   - C: image embeddings
   - D: vision-language model or fine-tuning
5. Stop compute when idle: `bash aws/99_stop_all.sh` and `uv run modal app list`.
