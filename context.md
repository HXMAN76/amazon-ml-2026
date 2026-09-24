# Amazon ML Challenge 2026: implementation context

Handoff notes for a new session (human or Claude). Written 2026-09-25 on branch `sai`. Companion documents: `handoff.md` (access, infra, commands, pitfalls, current state; read first), `plan.md` (approved baseline, evaluation and MLOps design), `research.md` (literature and findings), `code/business_entity_resolution/README.md` (how to run). Update the "Status" section as work proceeds.

## 1. Challenge

- **Task: Business Entity Resolution.** Three sources of noisy business records (`entity_id`, `business_name`, `business_address`, `country`). Source 1 is the deduplicated reference. For every S1 record, predict which S2 and S3 records are the same real business. An S1 entity may have zero, one or many matches.
- **Metric:** F0.5 computed per S1 entity, macro-averaged over all S1 entities. Precision counts double. A singleton (no true match) scores 1.0 for an empty prediction and 0.0 for any prediction.
- **Files are TSV** (tab-separated, read with `sep="\t"`). Ground truth: `source1_entity_id`, `matched_entity_ids` (comma-separated S2/S3 ids, empty for singletons).
- **Country is an open set.** Train has `US` and `India`. Test also has `France`, which is unseen in training. Do not hard-code or one-hot countries. Every test S1 entity must appear in the submission.
- **Outputs (both TSV):**
  - `matching_results.tsv`: the only file scored on the leaderboard.
  - `candidate_pairs.tsv`: the exact candidate set the model ran inference on. Every matched id must appear in it.
  - Rules: one row per S1 entity, no duplicate ids in a list, S2/S3 ids only, ids must exist in the test set.
- **Constraints:** final model MIT or Apache-2.0 licensed, at most 8B parameters. **No external lookups** (entity-resolution APIs, government registries, geocoding, internet augmentation): disqualification.
- **Limits:** 5 submissions per day, 3 days. Window 25 Sep 00:00 IST to 27 Sep 23:59 IST. Public leaderboard uses a subset of test; final ranking uses the private remainder. Ties go to the earlier submission.
- **Final package zip:** `output/` (both TSVs), `code/business_entity_resolution/{src,README.md,requirements.txt}`, and the filled `Documentation_template.md` (template is in S3 `docs/`).
- Official checker: `utils/validate_submission.py --matching ... --candidate ... --test-dir dataset/test` (in S3 `utils/`).

## 2. Data (S3)

`s3://ml-challenge-nooglers/ml-challenge-2026/raw/v1/` (bucket owned by a teammate's account; account A root and the notebook role can read and write it).

| File | Size |
|---|---|
| `dataset/train/train_source1.tsv` | ~200 MiB |
| `dataset/train/train_source2.tsv` | ~467 MiB |
| `dataset/train/train_source3.tsv` | ~480 MiB |
| `dataset/train/train_ground_truth.tsv` | ~121 MiB |
| `dataset/test/test_source1.tsv` | ~167 MiB |
| `dataset/test/test_source2.tsv` | ~486 MiB |
| `dataset/test/test_source3.tsv` | ~483 MiB |
| `docs/README.md`, `docs/Documentation_template.md`, `utils/validate_submission.py`, `student_resource.zip` | small / 1 GiB |

Observed from header peeks:
- Entity ids are random integers (`S1-925783039`), not sequential.
- India S2 names can be **Devanagari** (`राम मार्केटिंग प्राइवेट लिमिटेड`) while addresses are Latin and upper-case (`KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi`). Cross-script matching is required.
- US addresses have variable component order (`GREENSBORO, NC, 19 1/2 STARDUST TRAIL`). Some S3 names are domains (`wilfordhancock.com`), some addresses are empty.
- France records exist only in test (`63 R. DE DIEPPE, LILLE, Hauts-de-France`, names like `... Sarl`, `SCI ...`).
- Ground truth rows often list 3 to 5 matches across S2 and S3.
- Row counts (lines incl. header): train S1 2.21M, S2 5.03M, S3 5.29M, ground truth 2.21M rows (one per S1); test S1 1.73M, S2 4.89M, S3 5.08M. See section 2b for the measured structure.

## 2b. Measured data structure (stats3 job, train unless noted; full numbers, do not re-derive)

- Countries: train US 60%, India 40%. Test S1: India 810k, US 663k, **France 259k (15%)**; France is also 14% of test S2 and S3.
- **Match structure:** 5.6% of S1 are singletons (same in US and India). Mean 3.46 matches per S1 (max 11). 80.5% of S1 have both an S2 and an S3 match. S2 per S1 up to 5, S3 per S1 up to 6.
- **Exclusive ownership:** 7.64M matched ids, all unique, **no S2/S3 record is claimed by more than one S1**. A one-to-one constraint (each S2/S3 record goes to at most one S1) is valid on train. 73% of S2 and 75% of S3 records are matched; the rest (27%, about 0.8M US + 0.54M India in S2) are distractors.
- **Non-Latin text:** train S2 9.4% and S3 5.3% of names are non-Latin, all India; addresses non-Latin about 9%. Scripts seen: Devanagari, Telugu, Malayalam (so not only Hindi). S1 names are always Latin. Empty address: 3.4% in S2/S3, 0% in S1.
- **Generic names:** S1 has many repeats (`primary care group` 253 rows, 84 names with 100+ rows); S2/S3 have `primary care`, `urgent care`, `womens health` ~400 each. Names alone cannot decide; address does the work. Train S1 addresses are almost unique (max 14 repeats).
- **France differs:** test S1 has `bordeaux club sarl` (205 rows), `lille club sas` (122), and the same address repeated up to 101 times (`12 rue lyderic, lille, hauts-de-france`); test S2/S3 have tiny names (`cc`, `pc`, `lc` with 200-390 rows each) and repeated addresses (`27 rue jean bart, lille`). Many businesses share one address in France, unlike train. This is a real generalisation risk for any rule such as "same address means match".
- **Noise seen in matched groups:**
  - Names: HTML entities (`&amp;`, so run `html.unescape`), leetspeak/OCR (`C0mpany`, `5ecure`), typos (`Venmfes`), word-order shuffles, injected suffix words (`Indchem Power` matches `Indchem Center` and `Indchem Services`), domain forms (`ipower.com`, `www.cabreras.com`), DBA text (`Quoavi Co doing business as Asset Building Committee`), even a **completely different name at the same address** (`Ectozeph` matched to `Asset Building Committee`).
  - Addresses: missing components, dropped or altered digits (`B-59` vs `B-259`, `344` vs `1344`), inserted `Door No 467`, `CDP` glued to city (`CHICAGOCDP`), state abbreviations vs full names vs native script (`Telangana`, `TG`, `తెలంగాణ`), reordered components.

## 3. Accounts and infrastructure

Team of 4; account A is the lead's. Region for everything: **us-east-1**.

| Item | Value |
|---|---|
| AWS account (this session) | A, `567503593043`, Paid plan, $200 credits |
| CLI profile | `hxman-26` (root, created with `aws login`; re-run `aws login --profile hxman-26` when it expires) |
| Old hub bucket | `amlc-2026-hub-567503593043` (Modal/Kaggle pipeline on `main`, not used by this task) |
| Data bucket | `ml-challenge-nooglers` (note spelling: `challenge`) |
| Working bucket (own account) | `sagemaker-us-east-1-567503593043` (code sync, job queue, outputs) |
| SageMaker role | `sagemaker-competition-notebook-execution-role` (trusts sagemaker + ssm; `AmazonSSMManagedInstanceCore`, `AmazonSageMakerFullAccess`, inline `SageMakerSSHHelperControl`, `TeamS3Access`). Policy JSON in `iam/hxman/` |
| Notebook instance | `test-notebook`, `ml.g5.xlarge` (A10G 24 GB, 4 vCPU, 16 GB RAM), 100 GB volume, platform `notebook-al2023-v1` (AL2 platforms are rejected now), Docker 25.0.14 |
| GPU quota | `ml.g5.xlarge for notebook instance usage` = 1 (approved) |
| Lifecycle config | `autostop-1h`, source in `aws/notebook/onstart.sh` |

Cost: the g5 notebook bills about $1.01/h while InService, including idle. Stop it when not in use:

```bash
aws sagemaker stop-notebook-instance --notebook-instance-name test-notebook --profile hxman-26 --region us-east-1
```

Laptop is on a slow network. **Do not move big data through the laptop**; run everything on the g5 (decision: all real numbers come from the g5).

### How code runs on the g5 (no SSH needed)

1. Code lives in `code/business_entity_resolution/` in the repo. Sync it to S3 (always exclude `work/`, it is the data directory):
   `aws s3 sync code/business_entity_resolution s3://sagemaker-us-east-1-567503593043/ber/code --delete --exclude '*__pycache__*' --exclude '.pytest_cache/*' --exclude 'models/*' --exclude 'work/*'`
2. On boot the lifecycle script (`aws/notebook/onstart.sh`) installs a job-aware idle auto-stop and starts `aws/notebook/bootstrap.sh` in the background as `ec2-user`: it syncs the code to `/home/ec2-user/SageMaker/ber`, builds conda env `ber` (**Python 3.12**; 3.11 fails the pinned requirements), installs `requirements.txt`, writes `ber/.env_ready` only after imports succeed, and starts `jobrunner.sh`. (An earlier version wrapped a multi-line script in `sudo -i bash -c`, which collapsed the newlines into one command and silently did nothing.)
3. **Job queue:** put a bash script at `s3://sagemaker-us-east-1-567503593043/jobs/pending/<name>.sh`. The runner polls every 10 s, runs jobs one at a time in alphabetical order inside env `ber` from `/home/ec2-user/SageMaker` with `PYTHONPATH` set, **streams the log to `jobs/live/<name>.log` every 30 s**, and uploads the final log to `jobs/done/<name>.log` (last line `exit=<code>`). It holds `/tmp/job.lock` while a job runs so idle auto-stop does not kill it. Jobs must begin with `set -ex`, source conda, `conda activate ber`, and sync the code from S3 (they do not inherit the newest code otherwise).
4. Directories on the notebook: raw data `/home/ec2-user/SageMaker/dataset/{train,test}`; pipeline outputs `/home/ec2-user/SageMaker/work` (`BER_WORK`); code `/home/ec2-user/SageMaker/ber`. Never sync code with `--delete` without `--exclude 'work/*'`.
5. Watching a job from the laptop:
   `aws s3 cp s3://sagemaker-us-east-1-567503593043/jobs/live/<name>.log - --profile hxman-26` (running) or `.../jobs/done/<name>.log` (finished).
6. The job queue only trusts the account-A working bucket. Do not point the runner at the shared team bucket (other accounts could then run code on the instance).
7. Commands typed with `!`/`tail` in the desktop app terminal pane run on the laptop, not on the g5. To inspect the g5 directly, open Jupyter (presigned URL) and use its terminal.

### SSH Helper path (optional, from `remote-setup.md`)

Sai's SageMaker SSH Helper setup (local-mode container, `sm-ssh connect`). Laptop side is ready: `smssh-venv` (py3.12, `sagemaker==2.257.6`, `sagemaker-ssh-helper==2.3.0`, plus `botocore[crt]` needed for `aws login` creds), AWS CLI region patch applied, session-manager-plugin 1.2.835.0. `remote-ssh.remote.ipynb` was fixed (invalid JSON), uses `LOCAL_USER_ID=567503593043` and `instance_type='local_gpu'`. Caveat: the local-mode container is PyTorch 1.9.1 / py38, which is too old for modern libraries, so the job-queue path above is the working route.

## 4. Code: `code/business_entity_resolution/` (package `ber`)

Local dev env: `.venv-ber/` (Python 3.12, git-ignored). 14 tests pass (`make test`). The pipeline is a set of stages driven by a Makefile with hash-based caching (`make prepare sample block_eval`); every tunable lives in `configs/params.yaml`; runs are logged to MLflow (sqlite `work/mlflow.db`) and `work/runs/runs.jsonl`. Plan and design rationale: `plan.md`.

New pipeline (phase 0 and 1):

| Module | Role |
|---|---|
| `text.py` | deterministic normaliser: `html.unescape`, Latin-only accent stripping (Indic scripts kept), ASCII fast path (about 10 us per row), leetspeak repair in mixed letter/digit tokens, DBA/alias and domain-name splitting (`X dba Y`, `X | www.x.com`, `[www.x.com]`), legal forms into their own field, abbreviation expansion, `st`/`street`/`saint` share one token, glued `CDP` suffix, script fractions |
| `config.py`, `stamp.py`, `tracking.py` | params loader, Makefile stage stamps (hash of params sections and source files), MLflow + `runs.jsonl` logging |
| `stages/prepare.py` | TSV to Parquet per split and source (`rid`, raw and normalised columns) plus `labels.parquet` (`s1_rid`, `src`, `other_rid`); 24M records in about 6 to 9 minutes on 4 cores |
| `stages/sample.py` | 250k S1 training queries (seeded, folds 0..4), full S2/S3 pool kept |
| `stages/block.py` | DuckDB weighted token index. Token types: `n` name word, `a` address word/number, `p` 5-char prefix, `c` name-word x address-word, `m` name-word pair, `d` address-word pair, `h` house-number x address-word. IDF scoring, document-frequency cap, per-type rarest-token limits, top-K per S1, Parquet shards with resume, persistent pool-index cache |
| `stages/block_eval.py` | recall per configuration on a 20k-S1 subset: pair recall, S1 with all matches found, candidates per S1, recall by country, channel coverage, and a miss breakdown (unreachable / over df cap / lost to per-type limits and top-K) plus a no-cap lexical reachability diagnosis |
| `stages/pairs.py` | vectorised pair features (63): blocking scores per token type, rank/gap/margin inside the S1's list and the record's claimant list, rapidfuzz name and address similarities, name rarity counts, token coverage, glued-name, digit alignment, romanised and skeleton similarities |
| `stages/train_gpu.py`, `stages/predict.py`, `decision.py`, `validate.py` | XGBoost CUDA with grouped 5-fold OOF, exclusive assignment and threshold tuning; chunked prediction and TSV writing (never quote empty lists); local validator that parses raw lines like the official one |
| `src/scripts/qa_prepare.py`, `src/scripts/error_analysis.py` | normalisation samples; loss decomposition and error taxonomy of a trained model |

The legacy first baseline (dense per-country TF-IDF kNN, LightGBM) was removed from the package; `data.py` now only holds `read_tsv`.

Run on the g5 (via a queued job): `BER_DATA=/home/ec2-user/SageMaker/dataset BER_WORK=/home/ec2-user/SageMaker/work make prepare sample block_eval`.

## 5. Status (2026-09-25 about 04:30 IST)

Read `handoff.md` for access, commands and pitfalls. Summary:

Done:
- Infra: account A, notebook `test-notebook` (g5 A10G), S3 job queue with live logs, conda env `ber` (Python 3.12).
- Data profiled (section 2b). Phase 0 (normaliser, `prepare`, `sample`, Makefile, tests) complete.
- Blocking (token-index, DuckDB): pair recall 0.9416 at 30 candidates per S1 (US 0.970, India 0.899); 99.99% of true pairs share a token. Blocked all test S1 (51,892,359 pairs) and all train S1 (66,075,079 pairs).
- **v0** matcher (42 features, XGBoost CUDA): out-of-fold macro F0.5 0.9377. Test output passed the official validator (also with `--check-ids`); file at `s3://sagemaker-us-east-1-567503593043/runs/v0/output/`. Leaderboard score not yet known.
- Error analysis of v0 (`src/scripts/error_analysis.py`): matcher loss 4.0 points, blocking loss 2.2; weak spots were non-Latin names, empty addresses, look-alike distractors, missing name-rarity features (`research.md` section 12).
- **v1** matcher (63 features: name rarity and exact-name flags, token coverage, glued-name and digit-alignment features, romanised names via `anyascii` and consonant skeletons): out-of-fold macro F0.5 **0.9551** (+1.74 points); India 0.935, US 0.968, singleton entities 0.959; precision 0.989, recall 0.906. Loss now: blocking 2.19, matcher 2.30.
- Code zip for the portal built (`dist/business_entity_resolution_code.zip`, predates v1; rebuild before the final upload). `submission_checklist.md` maps every rule in the two PDFs to its status.

Running or pending: `v1b` (test features, predict, publish to `runs/v1/`), `v1c-blockeval` (recall of the new token types `g`, `x`, `k`). See `handoff.md` section 6.

Not done:
- Leaderboard scores for v0 and v1 (human uploads); decide which to keep.
- Blocking upgrade decision from `v1c-blockeval`, then re-block, re-featurise, retrain.
- Holdout protocol with bootstrap CI, calibration, record-level decision and per-S1 expected-F0.5, sibling features, France-proxy validation (US to India transfer).
- Vectorise the slow v1 features (about 60 s per 1.5M pairs).
- Methodology document from the template and the final package (team name, members, date still needed from the human).

## 6. Rules of the road

- Secrets, keys and login URLs never go into chat or git. `aws login` runs in the user's own terminal.
- Stop the notebook when idle. Never enable anything that forces the whole account into AWS Organizations.
- Keep every experiment logged with its OOF/validation score so submissions have a version history (the guidelines ask for it).
- `main` branch holds an older generic pipeline (Modal/Kaggle/EC2 scripts, hub bucket) from before the problem statement; it is not wired to this task.
