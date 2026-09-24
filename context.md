# Amazon ML Challenge 2026: implementation context

Handoff notes for a new session (human or Claude). Written 2026-09-25 on branch `sai`. Update the "Status" section as work proceeds.

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

## 2b. Measured data structure (stats3 job, train unless noted)

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

1. Code lives in `code/business_entity_resolution/` in the repo. Sync it to S3:
   `aws s3 sync code/business_entity_resolution s3://sagemaker-us-east-1-567503593043/ber/code --exclude '*__pycache__*' --exclude 'models/*'`
   The notebook pulls it to `/home/ec2-user/SageMaker/ber` on boot.
2. The on-start script builds conda env `ber` (python 3.11, `requirements.txt`) in the background (marker file `ber/.env_ready`) and starts `aws/notebook/jobrunner.sh`.
3. **Job queue:** put a bash script at `s3://sagemaker-us-east-1-567503593043/jobs/pending/<name>.sh`. The runner picks it up (polls every 10 s), runs it inside the `ber` env from `/home/ec2-user/SageMaker` with `PYTHONPATH=.../ber/src`, and uploads the log to `jobs/done/<name>.log` (ends with `exit=<code>`). Create `/tmp/job.lock` while a job runs so idle auto-stop does not kill it.
4. Data on the notebook: `/home/ec2-user/SageMaker/dataset/{train,test}` (synced from the data bucket by the stats job).
5. Code changes after boot: sync to S3, then a job script can `aws s3 sync s3://.../ber/code /home/ec2-user/SageMaker/ber` before running.
6. The job queue only trusts the account-A working bucket. Do not point the runner at the shared team bucket (other accounts could then run code on the instance).

### SSH Helper path (optional, from `remote-setup.md`)

Sai's SageMaker SSH Helper setup (local-mode container, `sm-ssh connect`). Laptop side is ready: `smssh-venv` (py3.12, `sagemaker==2.257.6`, `sagemaker-ssh-helper==2.3.0`, plus `botocore[crt]` needed for `aws login` creds), AWS CLI region patch applied, session-manager-plugin 1.2.835.0. `remote-ssh.remote.ipynb` was fixed (invalid JSON), uses `LOCAL_USER_ID=567503593043` and `instance_type='local_gpu'`. Caveat: the local-mode container is PyTorch 1.9.1 / py38, which is too old for modern libraries, so the job-queue path above is the working route.

## 4. Code: `code/business_entity_resolution/` (package `ber`)

Baseline v0, tested only on synthetic data (`pytest` passes 2 tests). Env for local dev: `.venv-ber/` (git-ignored).

| Module | Role |
|---|---|
| `normalize.py` | lowercase, punctuation, abbreviation expansion (Corp/Ltd/Pvt, Rd/St), legal-suffix removal (US/India/France forms), country aliases |
| `data.py` | TSV loading, normalised columns, ground-truth dict |
| `blocking.py` | top-K TF-IDF char n-gram kNN per country (core-name view and name+address view, unioned); `recall()` reports candidate recall |
| `features.py` | `Encoder` (4 TF-IDF spaces) and `pair_features` (49 features: rapidfuzz ratios, Jaro-Winkler, Levenshtein, Jaccard, number/PIN overlap, acronym match, TF-IDF cosines, per-S1 and per-candidate rank/gap) |
| `model.py` | LightGBM, 5-fold OOF grouped by S1 entity |
| `decide.py` | threshold and one-to-one selection tuned on OOF macro F0.5 |
| `metrics.py` | `f05_entity`, `macro_f05`, singleton/matched breakdown (verified against the 0.714 worked example) |
| `validate.py` | local format checker (the official script is authoritative) |
| `run.py` | `python -m ber.run train|predict` |
| `synth.py` | synthetic dataset generator for tests |

Run: `PYTHONPATH=src python -m ber.run train --data dataset --model models/v0` then `predict --data dataset --model models/v0 --out output`.

### Known problems with v0 at real scale (must fix before real numbers)

1. **Blocking does not scale.** It densifies chunks of a sparse matmul (`xa[chunk] @ xb.T`). With millions of S2/S3 rows on a 16 GB, 4 vCPU box this blows memory and time. Needs inverted-index/token or key blocking, and/or GPU dense kNN (torch or FAISS on the A10G, chunked top-k).
2. **`normalize.strip_accents` destroys Devanagari.** NFKD followed by dropping combining marks removes the matras. Only strip accents on Latin script; Devanagari needs transliteration or a multilingual embedding.
3. **Cross-script matching:** Devanagari S2 names versus Latin S1 names. Options: a multilingual embedding model (multilingual-e5 or bge-m3, both MIT) on the GPU, or an offline rule-based transliteration library (allowed: no external lookup). Embedding all records may take on the order of an hour on the A10G; measure first.
4. **Memory:** pandas plus per-row Python sets (`nums`) for millions of rows is too heavy. Use polars/arrow and compact columns, or process in country/region shards.
5. **Pair-feature loops:** Jaccard and acronym features use Python loops. Vectorise or move to shards.
6. Only 4 vCPUs on g5.xlarge; CPU-bound stages (rapidfuzz, LightGBM) will be slower than on the laptop's 24 cores. Weigh GPU-side work accordingly.
7. One-to-one selection is switched on only if it wins on OOF. Verify from the stats job how many S2/S3 records are claimed by more than one S1.

## 5. Status

Done:
- AWS account A set up for this task (role, notebook, lifecycle, working bucket, job queue). Old instances: none existed.
- Laptop SSH-helper client ready (optional path).
- Baseline v0 written and tested on synthetic data; code synced to `s3://sagemaker-us-east-1-567503593043/ber/code`.
- Notebook restarted with the new lifecycle config (was Pending at last check).
- Stats jobs done (`stats1`, `stats3`; `stats2` failed on a broken env). Env `ber` rebuilt on Python 3.12 (3.11 failed the pinned requirements). Dataset is on the g5 at `/home/ec2-user/SageMaker/dataset`. Findings in section 2b.

Not done:
- Redesign blocking and normalisation for scale and Devanagari (section 4).
- First real numbers on the g5: blocking recall ceiling, OOF F0.5.
- First submission early (ties go to the earlier submitter). Validate with the official script.
- Decide v1: dense embeddings (multilingual, MIT/Apache, at most 8B) for blocking and a cross-encoder or feature.
- Methodology doc from `Documentation_template.md`; final zip.
- Uncommitted on `sai`: `.gitignore`, `remote-ssh.remote.ipynb`, `code/`, `aws/notebook/`, `iam/hxman/`, this file.

## 6. Rules of the road

- Secrets, keys and login URLs never go into chat or git. `aws login` runs in the user's own terminal.
- Stop the notebook when idle. Never enable anything that forces the whole account into AWS Organizations.
- Keep every experiment logged with its OOF/validation score so submissions have a version history (the guidelines ask for it).
- `main` branch holds an older generic pipeline (Modal/Kaggle/EC2 scripts, hub bucket) from before the problem statement; it is not wired to this task.
