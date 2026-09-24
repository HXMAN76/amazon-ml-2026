# Handoff: Amazon ML Challenge 2026 (Business Entity Resolution)

Written 2026-09-25 about 03:00 IST, branch `sai`, repo `HXMAN76/amazon-ml-2026`. Audience: any other agent or person who must continue this work without the chat history. Read this first, then `context.md` (data facts, status), `plan.md` (approved design), `research.md` (literature and measurements), `code/business_entity_resolution/README.md` (how to run). Nothing here contains secrets; never add credentials to the repo.

## 1. One-paragraph state

Task: link each Source 1 (S1) business record to its S2/S3 records (entity resolution), scored by macro F0.5 per S1 entity; submissions are `matching_results.tsv` plus `candidate_pairs.tsv`. Window closes **Sun 27 Sep 2026 23:59 IST**, 5 submissions per day. Everything runs on one AWS SageMaker notebook (`test-notebook`, ml.g5.xlarge with an A10G, 4 vCPU, 15 GB RAM) driven from the laptop through an S3 job queue (no SSH needed). Pipeline stages built and run on real data: `prepare`, `sample`, `block` (test and train), `block_eval`. **A v0 submission chain is running**: `v0b` (pair features for train, then XGBoost GPU training) then `v0c` (test features, predict, validate, upload outputs to S3). No leaderboard score exists yet. Blocking recall is 0.942 (ceiling for v0) at about 30 candidates per S1.

## 2. Access and identity

| Item | Value |
|---|---|
| AWS account | `567503593043` (the lead's account A, Paid plan, $200 credits) |
| Region | `us-east-1` for everything |
| CLI profile | `hxman-26`, a **root** session created with `aws login --profile hxman-26`; it expires, then re-run that command in a terminal (browser sign-in). Verify: `aws sts get-caller-identity --profile hxman-26` |
| Laptop tools | `aws` CLI v2, `session-manager-plugin` 1.2.835.0 (installed, only needed for the optional SSH-helper path), `uv`, Python 3.12 |
| Local dev env | `/home/hxman/amazon-ml-2026/.venv-ber/` (Python 3.12; includes polars, duckdb, xgboost, mlflow, pytest). Create with `uv venv --python 3.12 .venv-ber && VIRTUAL_ENV=.venv-ber uv pip install -r code/business_entity_resolution/requirements.txt` |
| AWS MCP connector | not authorized in the desktop app; use the CLI instead |
| Shell gotcha | a command typed in the desktop app terminal pane (including `!cmd` style input) runs on the **laptop**. To run something **on the g5**, either queue an S3 job (section 4) or open Jupyter and use its terminal |

Buckets and resources:

| Resource | Name / value | Purpose |
|---|---|---|
| Data bucket (shared, owned by a teammate account) | `s3://ml-challenge-nooglers/ml-challenge-2026/raw/v1/` (`dataset/`, `docs/`, `utils/validate_submission.py`, `student_resource.zip`) | Raw data; account A root and the notebook role can read/write |
| Working bucket (account A) | `s3://sagemaker-us-east-1-567503593043` | `ber/code` (code sync), `ber/jobrunner.sh`, `ber/bootstrap.sh`, `jobs/{pending,live,done}/`, `runs/` (published artifacts) |
| Old hub bucket | `amlc-2026-hub-567503593043` | Old Modal/Kaggle pipeline on `main`; not used for this task |
| IAM role | `sagemaker-competition-notebook-execution-role` | trusts `sagemaker.amazonaws.com` and `ssm.amazonaws.com`; policies `AmazonSSMManagedInstanceCore`, `AmazonSageMakerFullAccess`, inline `SageMakerSSHHelperControl` and `TeamS3Access`; JSON in `iam/hxman/` |
| Notebook | `test-notebook`, `ml.g5.xlarge`, 100 GB volume, platform `notebook-al2023-v1` (AL2 platforms are rejected), direct internet on | The only compute |
| Lifecycle config | `autostop-1h` (source: `aws/notebook/onstart.sh`) | idle auto-stop after 1 h (skipped while `/tmp/job.lock` exists) and background bootstrap on every start |
| GPU quota | `ml.g5.xlarge for notebook instance usage` = 1 | approved |

Cost: about $1.01 per hour while the notebook is `InService`, including idle. Nothing else is running (no EC2, endpoints, training jobs).

Notebook control (all from the laptop):

```bash
export AWS_PROFILE=hxman-26 AWS_REGION=us-east-1
aws sagemaker describe-notebook-instance --notebook-instance-name test-notebook --query NotebookInstanceStatus --output text
aws sagemaker stop-notebook-instance  --notebook-instance-name test-notebook   # stops billing; disk persists
aws sagemaker start-notebook-instance --notebook-instance-name test-notebook   # lifecycle bootstrap restarts the job runner
# Jupyter login URL (short-lived, treat as a secret, never paste into chat or git):
aws sagemaker create-presigned-notebook-instance-url --notebook-instance-name test-notebook --query AuthorizedUrl --output text
```

After a stop/start the runner restarts automatically (bootstrap re-syncs code, checks the `ber` env, and starts `jobrunner.sh`); allow about 5 minutes.

## 3. Directory map

On the notebook (`/home/ec2-user/SageMaker`, persistent 100 GB volume):

| Path | Content |
|---|---|
| `dataset/{train,test}/*.tsv` | raw data (synced once from the data bucket) |
| `ber/` | code, synced from S3 by every job (`aws s3 sync ... --delete --exclude 'work/*'`) |
| `work/` (`BER_WORK`) | all pipeline outputs: `parquet/`, `sample/`, `blocks/`, `features/`, `models/`, `output/`, `runs/runs.jsonl`, `mlflow.db`, `.stamps`, `.done`, `official/`, `duckdb_tmp/` |
| conda env `ber` | Python 3.12 with `requirements.txt` installed |

DuckDB index files under `work/blocks/` are large (about 25 GB together); disk is 93 GB free at the start, so watch it (`df -h /home/ec2-user/SageMaker`).

In the repo:

| Path | Content |
|---|---|
| `code/business_entity_resolution/` | package `ber` (`src/ber`), `Makefile`, `configs/params.yaml`, `tests/`, `scripts/`, `requirements.txt`, `README.md` |
| `aws/notebook/` | `onstart.sh`, `bootstrap.sh`, `jobrunner.sh` (the infra scripts, also copied to S3 `ber/`) |
| `iam/hxman/` | IAM policy documents for the role |
| `context.md`, `plan.md`, `research.md`, `handoff.md` | project documents |
| `remote-setup.md`, `remote-ssh.remote.ipynb`, `iam/*.json` | the SageMaker SSH-helper path from a teammate (optional, superseded by the job queue) |
| `main` branch | older generic pipeline (Modal, Kaggle, EC2 scripts); not wired to this task |

## 4. Running code on the notebook: the job queue (the main workflow)

Protocol:
1. Put a bash script at `s3://sagemaker-us-east-1-567503593043/jobs/pending/<name>.sh`.
2. The runner (`aws/notebook/jobrunner.sh`, started at boot) polls every 10 s and runs pending jobs **one at a time in alphabetical order** inside `conda activate ber` from `/home/ec2-user/SageMaker`.
3. It streams the log to `jobs/live/<name>.log` every 30 s while running and uploads the final log to `jobs/done/<name>.log`; the last line is `exit=<code>`. While a job runs it holds `/tmp/job.lock`.
4. Jobs do **not** inherit newest code automatically: every job script starts by syncing it.

Standard job header (copy exactly; `set -ex` makes failures visible):

```bash
set -ex
source /home/ec2-user/anaconda3/etc/profile.d/conda.sh; conda activate ber
aws s3 sync s3://sagemaker-us-east-1-567503593043/ber/code /home/ec2-user/SageMaker/ber --delete --exclude 'work/*' --only-show-errors
cd /home/ec2-user/SageMaker/ber
export BER_DATA=/home/ec2-user/SageMaker/dataset BER_WORK=/home/ec2-user/SageMaker/work
# ... your commands, e.g.  python -m ber.stages.block --split test
```

Update and run workflow (from the repo root on the laptop):

```bash
export AWS_PROFILE=hxman-26; B=sagemaker-us-east-1-567503593043
# 1. edit code, run tests locally
(cd code/business_entity_resolution && ../../.venv-ber/bin/python -m pytest -q tests)
# 2. publish code (always exclude work/, it is the data directory on the notebook)
aws s3 sync code/business_entity_resolution s3://$B/ber/code --delete --exclude '*__pycache__*' --exclude '.pytest_cache/*' --exclude 'models/*' --exclude 'work/*' --only-show-errors
# 3. submit a job
aws s3 cp my_job.sh s3://$B/jobs/pending/my_job.sh
# 4. watch (live while running, done when finished)
aws s3 cp s3://$B/jobs/live/my_job.log - | tail -30
aws s3 cp s3://$B/jobs/done/my_job.log - | tail -60
aws s3 ls s3://$B/jobs/ --recursive        # what is pending / live / done
```

Notes that cost time before:
- A job name that already has a log in `jobs/done/` is not overwritten until the new run finishes; use a fresh name for reruns (`v0c`, `v0c2`, ...).
- The runner lists pending jobs once per loop; alphabetical order decides the order (`v0a` < `v0b` < `v0c`).
- Killing a running job needs a shell **on the notebook** (Jupyter terminal): `pkill -f ber.stages.<name>`; the queue then proceeds.
- The runner is replaced by uploading a new `ber/jobrunner.sh` to S3 and running a job like `swaprunner` (waits 60 s, kills the old runner, starts the new one). The last swap added the live-log streaming.
- Never point the queue at the shared team bucket: whoever can write there could run code on the instance.
- `aws s3 sync ... --delete` **without** `--exclude 'work/*'` would delete pipeline outputs under `ber/work`; outputs now live in `SageMaker/work`, one level above `ber/`, but keep the exclude.
- Long jobs: the idle auto-stop is job-aware; the notebook will not stop while `/tmp/job.lock` exists.

Optional interactive access: open Jupyter through the presigned URL, then File → New → Terminal. GPU check: `nvidia-smi`. The notebook host has an A10G (23 GB), CUDA 13.2, Docker 25.

## 5. Pipeline and how to run each stage

Everything is in `code/business_entity_resolution`; parameters in `configs/params.yaml`; the Makefile caches by hash of params and source. Run on the notebook with the env vars from the job header.

| Stage | Command | Result | Measured on the g5 |
|---|---|---|---|
| prepare | `make prepare` | Parquet for both splits and `parquet/train/labels.parquet` (7,638,365 true pairs) | about 6 to 9 min for 24M records |
| sample | `make sample` | `sample/train_s1.parquet`: 250,000 S1 with folds 0..4 | seconds |
| block_eval | `make block_eval` | `blocks/eval_report.json` (recall per configuration) | index cached; runs seconds each at cap 800 |
| block | `python -m ber.stages.block --split test` and `--split train --all-train` | `blocks/{split}/cand_*.parquet` | test 51.9M pairs in 18.5 min (includes pool index build), train 66.1M pairs in 13 min |
| pairs | `python -m ber.stages.pairs --split train` (7.48M pairs) / `--split test` (51.9M) | `features/{split}/part_*.parquet` (45 columns) | train blocking stats 46 s, string features about 13 s per 1.5M pairs |
| train_gpu | `python -m ber.stages.train_gpu --name v0` | XGBoost (CUDA) 5-fold grouped OOF, exclusive assignment and threshold tuning; `models/v0/{xgb.json,config.json,report.json,oof.parquet}` | pending |
| predict | `python -m ber.stages.predict --name v0` | `output/v0/{matching_results.tsv,candidate_pairs.tsv}` and validation (official validator if found at `work/official/validate_submission.py`) | pending |

Token types in blocking: `n` name word, `a` address word, `p` 5-char prefix, `c` name x address word, `m` name-word pair, `d` address-word pair, `h` house-number x address word. Key parameters: `k` 30, `cap_df` 800 (tokens more frequent than this in the 10.3M pool are ignored; raising it changes nothing but costs up to 100x time), `per_type` rarest-token limits. Candidate id convention: `pid = src * 10_000_000 + rid` (`src` 2 or 3). S1 identifier is `rid`, the row index in the Parquet (0-based, equals row number in the TSV).

Local tests: `make test` or `pytest -q tests` in `code/business_entity_resolution` (16 tests including an end-to-end synthetic run of the whole v0 chain; xgboost falls back to CPU when no GPU).

## 6. What is running or pending right now

| Job | State at the time of writing | Notes |
|---|---|---|
| `v0a` | done | blocking for test (51,892,359 pairs) and all train S1 (66,075,079 pairs) |
| `v0b` | running | `pairs --split train` (about 5 chunks of 1.5M) then `train_gpu --name v0` |
| `v0c` | pending (already uploaded) | `pairs --split test`, `predict --name v0`, copies the validator, then publishes `matching_results.tsv`, `candidate_pairs.tsv`, model files and `runs.jsonl` to `s3://sagemaker-us-east-1-567503593043/runs/v0/` |

After `v0c` finishes: fetch `runs/v0/output/matching_results.tsv` to the laptop, run the official validator locally (`python3 <validator> --matching ... --candidate ... --test-dir <test dir>`; the validator is at `s3://ml-challenge-nooglers/ml-challenge-2026/raw/v1/utils/validate_submission.py` and needs the three test TSVs; alternatively read the validator result from the `v0c` log), then upload `matching_results.tsv` in the Unstop portal by hand (leaderboard uploads are done by the human; do not automate portal actions). Record the leaderboard score next to the run in `runs.jsonl` notes.

## 7. Results and facts to trust (do not re-derive)

- Data structure and noise: `context.md` section 2b (row counts, singleton rate 5.6%, mean 3.46 matches per S1, exclusive ownership, France differences, noise catalogue).
- Blocking, 20k train S1 vs the full 10.3M pool, cap 800, K 30, all token types: **pair recall 0.9416** (US 0.970, India 0.899), 29.9 candidates per S1, S1 with all matches found 0.8375. Misses: 2.2% over the df cap, 3.7% lost to per-type limits and top-K. Diagnosis with no cap: only 0.01% of true pairs share no token; the rarest shared token has df at most 100 for 96%, at most 800 for 97.9%. So lexical blocking is enough; a dense channel is only for the non-Latin India minority (about 9%).
- Higher `cap_df` does not help (0.9403 at 5000, 0.9412 at 20000) and is very slow (67 s and 797 s vs 5 s).
- Normaliser throughput: about 10 microseconds per row after the ASCII fast path (was about 400 before).

## 8. Known pitfalls (each cost real time)

1. Makefile `WORK ?= work` once overrode `BER_WORK`, silently writing under `ber/work`; fixed (`WORK ?= $(or $(BER_WORK),work)`).
2. Multi-line `sudo -u ec2-user -i bash -c "..."` collapses newlines into one command; use a script file (`bootstrap.sh`).
3. conda Python 3.11 fails the pinned requirements (scipy/pandas versions); the env must be **3.12**. `.env_ready` is now written only after imports succeed.
4. `aws login` credentials need `botocore[crt]` inside Python venvs; the CLI itself is fine.
5. Notebook platform `notebook-al2-v2/v3` are rejected; only `notebook-al2023-v1` works.
6. `~/onstart.log` on the notebook may be root-owned from old runs; use new log names.
7. Polars: `str.split` etc. emit deprecation warnings (harmless); `pl.concat(how="horizontal")` is deprecated (use `hstack`).
8. DuckDB pool index cache is keyed by an index-parameter hash; changing `prefix_len`, `min_prefix_len` or `comp` rebuilds it (about 4 min); changing `cap_df`, `per_type`, `k` does not.
9. Job scripts must sync code themselves; forgetting it runs stale code.
10. `s3 sync --delete` needs `--exclude 'work/*'` when the target contains work data.

## 9. Design summary and next steps

Approved design is `plan.md` (multi-channel blocking, feature matcher, calibration, exclusive assignment, expected-F0.5 per-S1 decision, Makefile + MLflow, single account, baseline first). A teammate proposed a **record-centric** v1 (each S2/S3 record picks its owner or none; sibling consensus; fine-tuned multilingual bi-encoder; Modal); review conclusions: adopt the record-level decision with a none class and calibrated owner probability, the forensics of noise operators from matched train pairs, the evaluation discipline (locked holdout, bootstrap CI, US to India transfer as France proxy, adversarial train-vs-test validation, an empty submission to measure the test singleton share), and sibling features as second-stage stacking; postpone dense-first retrieval, FAISS-GPU, Modal and the fine-tuned e5 until v0 shows where India is weak. Do not rebuild `prepare`, `block` or `features`: extend them.

Suggested order after the v0 submission:
1. Holdout protocol and bootstrap CI (extend `sample.py` to a dev/holdout split; reuse `decision.macro_f05`).
2. Record-level decision: softmax or GBM over each record's candidate S1 list plus a none option, calibrated with isotonic regression; per-S1 expected-F0.5 prefix selection by dynamic programme; vetoes on conflicting PIN or house number.
3. Raise K (60 to 100) after a cheap first-stage ranker prunes candidates before expensive string features.
4. Forensics report from matched train pairs to improve the normaliser (leet mappings, abbreviations, domain constructions); rerun `prepare` (9 min) only when `text.py` changes.
5. Dense multilingual channel (bge-m3 or multilingual-e5, both MIT) only for non-Latin India names if India recall stays below US.
6. Cross-encoder on the ambiguous band only, after a real score exists.
7. Freeze, reproduce from scratch, methodology write-up from `docs/Documentation_template.md` in S3, submission zip layout in `context.md` section 1.

## 10. Rules

- No secrets, keys, presigned URLs or login links in git or chat. `aws login` is run by the human in their own terminal.
- No external data lookups (entity-resolution APIs, registries, geocoding). Pretrained models must be MIT or Apache-2.0 and at most 8B parameters; log licences.
- Portal uploads and any action with side effects outside the account (submissions, billing changes, deleting data) are done or approved by the human.
- Stop the notebook when idle for long periods.
- Log every experiment (`runs.jsonl` plus MLflow) with its validation number so the submission history required by the guidelines exists.
- Commit documentation and code changes to branch `sai`; run tests before syncing to S3.
