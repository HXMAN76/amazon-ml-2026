# Handoff: Amazon ML Challenge 2026 (Business Entity Resolution)

- Archived docs (26 Sep): `docs/archive/` holds v1, v2, research-v2, plan, the teammate handoff, remote-setup and the SSH-client requirements; do not follow them.

Written 2026-09-25, refreshed 25 Sep about 14:00 IST, branch `sai`, repo `HXMAN76/amazon-ml-2026`. Audience: any other agent or person who must continue this work without the chat history. Read this first, then `context.md` (data facts, status), `plan.md` (approved design), `research.md` (literature and measurements), `code/business_entity_resolution/README.md` (how to run). Nothing here contains secrets; never add credentials to the repo.

## 0.1 Live status (26 Sep 2026, 16:30 IST; newest, wins over everything below)
- **The portal gap is France.** Portal probes: France only 0.187, US only 0.453. France F0.5 is about 0.92 (range 0.90 to 0.95), US and India about 0.991 (holdout-consistent). France costs about 0.010 of score, the whole 0.0093 gap. The model believes France is fine (0.987): it over-claims there (63.8% of the pool claimed against 59% in the US, 3.53 matches per S1 against 3.40, 14 times more S1 with impossible counts of more than 5 S2 matches). Full analysis: `research.md` sections 23 and 24; plan with gains: `EXPERIMENTS.md` section 10.
- **Best model `s22`**: holdout 0.99054 (+0.00029 over `s17`, paired CI [0.00018, 0.00039]); files `runs/s22/output/`. France-threshold variants of `s22` for the portal: `runs/s22f97`, `runs/s22f985` (planned for today's last submission), `runs/s22f995`. Submissions used on 26 Sep: 12:31 `s17` 0.981, 15:56 `s17pf` 0.187, 15:57 `s17pu` 0.453; one left today.
- **Running**: `s24` (Qwen3-0.6B third cross-encoder on the second notebook, lane B3: `rn1` training to about 22:00, `rn2` scoring, `rn3` stack; result about 02:00), `s25` (address-multiplicity features, main lane 1, `rs1`, about 16:45), `s20` (e5-large cross-encoder, lane B2: `ry1`, then a stack), `s19` (`v8`, lane B: `rx3`, `rx4`). Finished: `s21` (0.99000, -0.00024), `s23` (0.98955, -0.0007; distractor density is not the cause), `s18` (no gain).
- **Analysis scripts added** (`src/scripts/`): `attrition.py`, `attrition2.py`, `ambiguity_profile.py`, `poststrat.py`, `france_profile.py`, `france_profile2.py`. Results in `research.md` sections 23 and 24.
- **Do not repeat**: extra empty-address neighbours (`v6`, `s7`), iterated consensus (`s9`), calibrated expected-F0.5 decisions, small cross-encoder alone (`s18`), test-like-universe training (`s23`), S1-attribute reweighting.
- **Next**: read the France curve from the portal (`s22f985` today, `s22f97` and `s22f995` tomorrow), then a France calibration by distribution matching and France-aware features. Freeze on Sunday: zip, official validator with `--check-ids`, methodology document (still describes older numbers).

## 0. Latest status (25 Sep 2026, about 23:20 IST; where this differs from sections 1 and 6, this section wins)

**Leaderboard (public subset), in order of submission**

| File | Submitted | Portal score | Locked-holdout F0.5 (150k train S1) |
|---|---|---|---|
| `v2` | 25 Sep 13:46 | 0.944 | 0.9565 |
| `s4` | 25 Sep 22:21 | **0.953** | **0.9708** |
| `s3all` | 25 Sep 22:53 | 0.949 | 0.9677 |

`s2` was never uploaded. Both `s4` and `s3all` are at `runs/<name>/output/` and passed the official validator (`s4`: without `--check-ids` on the notebook, plus `check_submission.py`, which does the ID checks). Version history for the methodology document: v2, s4, s3all, then whatever is uploaded on 26 and 27 Sep.

**Update 26 Sep 2026 about 13:00 IST.** **Portal: `s17` scored 0.981** (12:31; holdout 0.99025, gap 0.0093; `s12` 0.972 at 09:49). Ladder and all experiments: **`EXPERIMENTS.md`** (read it first: score ladder, what runs on every GPU, untried ideas, a table for teammates' results). New findings: adversarial validation of the train to test shift gives **AUC 0.8716** (drivers: blocking-score scale, competition and name-count features, which depend on the size of each split's S1 and pool sets), so shift-robust variants are running: `v8` (joint name counts), `v9` (first stage without blocking-score and competition features) and `s22` (cross-encoder on every shortlisted pair). Compute: main notebook `ml.g5.16xlarge` (lanes `jobs`, `jobs2`), second notebook `test-notebook-2` `ml.g5.24xlarge` with 4 A10G and lanes `jobsB`, `jobsB2`, `jobsB3` (`jobsB4` free); `ml.g5.12xlarge` had no capacity. Uploads for today: `s17` and `s17f85` (France threshold 0.85, `runs/s17f85/output/matching_results.tsv` with `s17`'s candidate file).

**Update 26 Sep 2026 about 12:10 IST.**
- **Portal:** v2 0.944, s3all 0.949, s4 0.953, **s12 0.971976 (rank 402)**; s12 holdout 0.98505, gap 0.0131 (s4 had 0.0178). The portal moved +0.019 from s4 to s12 while the holdout moved +0.0143, so the dense channels narrowed the gap. Nothing after s12 has been uploaded yet.
- **Best models by holdout F0.5:** s14 **0.98986** (base `v5`, cross-encoder 2 = `multilingual-e5-base` fitted on 700k S1, band 0.01 to 0.99; +0.00069 [0.00054, 0.00083] over s13), s16 0.98946 (base `v7`, cross-encoder 1, competition features on the refined probability `--xcons`), s15 0.98935, s13 0.98917. In flight: `s17` (base `v7`, cross-encoder 2 rescored on `v7`'s bands, cross-encoder 1 as `xs2`, `--xcons`), then `s18` (small cross-encoder on the base model's data, to separate model size from data size).
- **Remaining loss of s15 on the holdout:** oracle on our candidates 0.9957; blocking 0.0043; matcher 0.0064; precision 0.9980; recall against all true pairs 0.9718. 74% of the still-missed true pairs (10.8k of 14.6k) are pool records with an EMPTY address (4.4% of true pairs, 75% reach the candidates, 53% matched): name-only records, mostly ambiguous. The recoverable loss from bigger encoders or more data is small (about 0.002); the main lever is the holdout to portal gap.
- **Why the portal is lower than the holdout (estimates, `research.md` section 21):** the test is harder (5.8 pool records per S1 against 4.7; about 40% of pool records unowned against 26%) about 0.003; country mix about 0.001; France 0.001 if the model is right; public-subset noise about 0.002; unexplained about 0.008 (France miscalibration or covariate shift). Model-based test estimates for s15: US 0.9940, India 0.9951, France 0.9866 (uncertain best candidate 0.5% in France against 0.2% and 0.1%; 3.60 mean matches per S1 against 3.43 and 3.40).
- **New tools:** `src/scripts/reemit.py NAME NEWNAME --thr france=0.85` (per-country thresholds from a model's `pair_p`), `adv_validation.py` (holdout versus test pairs, adversarial AUC and drifting features), `country_expected.py`, `noise_analysis.py`, `paired_models.py`; stack options `--xenc --xenc-dir --xenc-dir2 --xcons --xenc-fit-more --decoy --extra --sub-q --tag --set`; `pairs.joint_counts` (name-rarity counts over train and test together, off by default; the S1 count differs by 21% between train and test).
- **Infrastructure:** the main notebook is `ml.g5.16xlarge` (queue prefixes `jobs/`, `jobs2/`); a second notebook `test-notebook-2` (`ml.g5.24xlarge`, 4 A10G, 96 vCPU, 250 GB disk; `ml.g5.12xlarge` was out of capacity) with prefixes `jobsB/`, `jobsB2/` (chosen by name in `aws/notebook/bootstrap.sh`); state for it is exported by job `rz1` to `s3://sagemaker-us-east-1-567503593043/state/`; heartbeats in `diag/<notebook>.txt`. The AWS CLI login expires after about 5 hours (26 Sep 07:30): re-run `aws login --profile hxman-26`.
- **Backlog for tomorrow (more submissions may be allowed):** per-country probe files (empty one country's predictions from a good file; the score drop gives that country's F0.5 on the test set); France threshold variants (`reemit.py`); a threshold check for the higher distractor rate; cap of 5 S2 and 6 S3.

**Update 26 Sep 2026 about 06:40 IST.** Best model **`s15`**: holdout F0.5 **0.98935** (stack on the first stage `v7` with the cross-encoder score, decoy edit features, extra carried columns, 1.5M S1, depth 9; 4.74 candidates per S1 on test; files `runs/s15/output/`). Trail on the holdout: `s6` 0.98308, `s8` 0.98362 (decoy+extra), `s12` 0.98505 (+1.5M S1, depth 9), `s11` 0.98887 (+cross-encoder), `s13` 0.98917, `s15` 0.98935. Portal scores are still only v2 0.944, s3all 0.949, s4 0.953: **the newest models (s5 onward) have never been uploaded, which is the main open question** (holdout to portal gap has been 0.012 to 0.018). New stages: `stages/xenc.py` (cross-encoder), `train_gpu --extra N --set ...` (first stage on more S1), stack options `--decoy --extra --xenc --xenc-dir --xenc-fit-more --sub-q --tag --set`; `aws/notebook/jobrunner.sh` now has two lanes (`jobs/` and `jobs2/`) and a per-minute diagnostic heartbeat (`s3://sagemaker-us-east-1-567503593043/diag/latest.txt`). Notebook is `ml.g5.16xlarge` since 02:11 (restarted 04:12 and 06:00 after two unexplained stalls about 58 minutes after each boot; see the pitfalls in section 4). In flight: `rp1`/`rp2` (stronger cross-encoder `multilingual-e5-base`, stack `s14`). Full detail: `ARCHITECTURE.md` (section 3.0 and the version table).

**Update 26 Sep 2026 about 02:00 IST.** New best files: `s5` (holdout 0.9832, India 0.9812, US 0.9845; stack on `v5` with the name+address dense channel `dense_all`, `runs/s5/`) and `s6` (same score, candidate shortlist keeps 4.9 per S1, 126 MB candidate file, `runs/s6/`). `s5`/`s6` portal scores are not known yet. Running: `zt3` and `zt4` (`v6`, `s7`: extra dense neighbours for empty-address pool records). Planned: notebook to `ml.g5.16xlarge` ($5.12/h, 64 vCPU) after `zt4`, credit budget 270, stop the notebook after the last heavy job. The teammate-facing summary of the architecture, the AWS how-to and the optimization list is **`TEAM_GUIDE.md`** in the repo root; the complete architecture and version history (every version from `v0` to `s6`, what was tried and dropped) is **`ARCHITECTURE.md`**. `v6`/`s7` (extra neighbours for empty-address pool records) gave no gain (0.98309 against 0.98308) and were dropped. Notebook switched to `ml.g5.16xlarge` at 02:09 IST on 26 Sep (budget 270 credits, $5.12/h; stop it after the last heavy job); two job lanes (`jobs/` and `jobs2/`).

**Model lineage (all numbers on the locked holdout unless stated)**

| Model | What changed | Holdout F0.5 | Paired gain |
|---|---|---|---|
| `v2` | 63 features, cascade blocking (K 100 raw, best 30 by a learned ranker) | 0.9565 [0.9559, 0.9572] | reference |
| `s1` / `s2` | consensus stacking; `s2` adds house-number relation and digit-consensus features | 0.9617 / 0.9671 | +0.0051 / +0.0106 over v2 |
| `s3all` | adds TF-IDF cosine and name/address consensus features; ablations: no consensus 0.9676, no TF-IDF 0.9672 | 0.9677 | +0.0111 [0.0108, 0.0115] over v2 |
| `v3` | first stage with the name-only dense channel (multilingual-e5-small fine-tuned on 92k India S1; top-5 non-Latin candidates added); France address rules; index version 4 | 0.9598 [0.9592, 0.9604] (out-of-fold 0.9602 vs 0.9568) | +0.0033 over v2 |
| `s4` | stack on `v3` (S1 used to fine-tune the encoder excluded from stack training) | **0.9708** | +0.0031 [0.0028, 0.0034] over `s3all`; +0.0037 [0.0034, 0.0040] over `s2` (`src/scripts/paired_models.py`) |
| `e1` | calibration plus expected-F0.5 selection | null result (+0.0001, CI includes 0), not shipped | |

Portal gaps: v2 0.0125, s3all 0.0187, s4 0.0178 (holdout minus portal). Dense plus France rules moved the portal +0.004 (holdout +0.0031), so those changes are real. The stacking layers transferred only about half (holdout +0.011, portal +0.005).

**What is still lost (error analysis of `s4`, `src/scripts/error_analysis.py`, `miss_analysis.py`)**
- Holdout loss versus the oracle on the candidates: blocking 0.0166 (was 0.0195), matcher 0.0126 (was 0.0188). India 0.9569, US 0.9802.
- 24,230 of 518,468 true holdout pairs (4.67%) are never proposed: pool record with empty address 7,991 (33%), typos or scrambled names 5,749 (24%), non-Latin pool name 5,585 (23%), glued or domain-style name 4,905 (20%). Of the non-Latin misses only 17.8% appear in the first dense channel's top-10.
- Remaining false positives are look-alike distractors; false negatives are mostly empty-address or suffix-noise pairs.

**New candidate channel `dense_all` (stages/dense_all.py, params section `dense_all`)**
- Encodes "name | address" of every S1 and every S2/S3 record with multilingual-e5-small fine-tuned on 250k S1 (all countries, one mined hard negative per pair, 756 s), retrieves for every pool record its nearest S1 records (pool to S1, because a record has one owner).
- Holdout report (`zn2`): the top-1 neighbour with no cosine cut-off adds 1.09M pairs (0.5 per S1) and recovers 16,025 of 24,230 missed pairs (66%, 3.1% of all true pairs); top-3 adds 19.6M pairs, top-5 39M for little more; any cosine cut-off discards most of the gain. Setting: `k_merge 1`, `tau 0`.
- Merged into the shards: test +1,452,586 pairs, train +1,094,973. New feature columns `dall_cos`, `dall_rank`. Merge is idempotent (backup `blocks/{split}_predall`).
- `v5` (first stage on these candidates, 67 features): out-of-fold F0.5 **0.9757** (v3 0.9602), top feature `dall_cos`. Holdout (jobs `zp3`) and the stack `s5` (`zp4`) are pending: ship only if the paired interval against `s4` excludes 0. Caution: test top-1 cosine is lower (0.795 test, 0.825 train) and test has more unowned pool records (about 40% against 26%), so the new feature may shift on the test set.

**Country mix and France (nothing is labelled for France)**
- Test S1: US 663,106 (38.3%), India 809,986 (46.8%), France 259,452 (15.0%); train is 60% US and 40% India. Re-weighting the holdout to the test mix lowers `s4` by about 0.0035 (US 0.9802, India 0.9569).
- `src/scripts/country_expected.py` estimates F0.5 from the model probabilities (bias measured on the labelled holdout: US +0.0068, India +0.0279). Test estimates for `s4`: France 0.9801, US 0.9824, India 0.9795; the test is harder than train for every country (US estimate 0.9871 on holdout, 0.9824 on test; test has 5.8 pool records per S1 against 4.7). Corrected for that and for the mix, `s4` lands near 0.965 against the portal 0.953; about 0.012 stays unexplained (France miscalibration, public-subset noise, or unmodelled misses).
- France output looks healthy: matched 94.8% (US 94.3%, India 93.4%), mean matches 3.45 (US 3.42), 62% of France pool records owned (US 59%, India 55%), mean name similarity of predicted pairs 90.5 (US 91.2) but twice the share of very weak names (3.6% below 40 against 1.9%); S1 sharing an address with 5 or more others (6,900 in France) average about 3.0 matches instead of 3.46.

**New organiser rule (email, 25 Sep evening): `candidate_pairs.tsv` counts toward the final ranking and a smaller candidate set per S1 ranks higher.** Our sets average about 32 per S1 (35 for India). Plan: a shortlist stage by first-stage probability inside the pipeline, so the stack trains and scores only the shortlist and `candidate_pairs.tsv` lists exactly what the final model scored (never trim the file after scoring). Job `zo8` (`src/scripts/shortlist_eval.py s4 v3`) measures the F0.5 cost of K = 3 to 20 by first-stage rank. Also pending: cap of at most 5 S2 and 6 S3 matches per S1 (343 and 96 S1 of the `s3all` output exceed it, worth at most +0.0002).

**Jobs now (one notebook, sequential, alphabetical)**: `zp3` (v5 holdout, test features and predict, checker, publish `runs/v5/`), `zo8` (shortlist evaluation), `zp4` (stack `s5`, checker, paired test against `s4`, publish `runs/s5/`). Expected finished about 01:30 to 02:30 IST on 26 Sep. Disk 26 GB free. The AWS login expires periodically: `aws login --profile hxman-26`.

**Plans for 26 and 27 Sep (window closes Sun 23:59 IST, 5 submissions per day)**
1. Read `zo8` and choose K; implement the shortlist in the stack build and predict, retrain.
2. Gate `s5`; upload the best holdout model. If the test distractor rate shifts the optimum, try the same model at two or three thresholds (a single parameter) with the portal.
3. If time: decoy edit features (substitution versus indel, Hamming distance, length change) in the stack and US/Indian state-name normalisation (`Tennessee` and `TN`; teammate's v5 proposal), each behind its own paired gate; cap 5 S2 and 6 S3.
4. Freeze: rebuild the code zip, one clean reproduction, final validator (`--check-ids` needs test_source2/3 and is slow from the laptop: run it on the notebook), methodology document (`submission/Documentation_template.md`), refresh these documents.

**Teammates:** the v4 plan (cross-encoder on the uncertain band, dense blocking with an English-only encoder) and a v5 branch built on `sai` at `1688d18` (state names, exact keys, sibling expansion, cross-encoder, decoy features, per-country thresholds, cap) are reviewed in `research.md` section 19. The user decided to leave the teammate's work separate. Warning: the v5 plan reuses our model names `v5` and `s5`; on the shared notebook or S3 code prefix that would overwrite our runs, so they must use other names and prefixes.

## 1. State in brief

Task: link each Source 1 (S1) business record to its S2/S3 records (entity resolution), scored by macro F0.5 per S1 entity; submissions are `matching_results.tsv` plus `candidate_pairs.tsv`. Window closes **Sun 27 Sep 2026 23:59 IST**, 5 submissions per day. Everything runs on one AWS SageMaker notebook (`test-notebook`, ml.g5.xlarge with an A10G, 4 vCPU, 15 GB RAM) driven from the laptop through an S3 job queue (no SSH needed).

- **v0** (42 features, XGBoost on GPU): out-of-fold macro F0.5 **0.9377**. Output at `s3://sagemaker-us-east-1-567503593043/runs/v0/output/`.
- **v1** (63 features: name rarity, exact-name, token coverage, glued-name, digit alignment, romanised names via anyascii, consonant skeletons): out-of-fold macro F0.5 **0.9551** (India 0.935, US 0.968, singleton entities 0.959, precision 0.989, recall 0.906). Output at `runs/v1/output/`. **Both v0 and v1 passed** the official validator (including `--check-ids` on both v1 files) and the bundled rule checker; on test, France is matched at 94.8% of S1 (US 94.3%, India 92.9%). The human uploads to the portal; leaderboard scores are not yet known to the agent. v1 is the recommended upload.
- **v2** (v1 features plus cascade blocking, K 100 raw candidates pruned to 30 by a learned first-stage ranker): out-of-fold macro F0.5 **0.9568**; locked holdout (150k train S1 never seen by the model, seed 2026) **0.9565, 95% CI [0.9559, 0.9572]**, precision 0.990, recall 0.909. Test output at `runs/v2/output/`, passes the official validator and the rule checker (France 94.9% matched). **Leaderboard: v2 scored 0.944** (public subset, first portal submission, 25 Sep 1:46 PM IST). The gap to the holdout (0.0125) is the cost of France having no training labels plus the public-subset difference.
- **Remaining loss (4.5 points on the train sample):** blocking recall 2.19 (pair recall 0.941) and matcher 2.30.
- **Blocking upgrade result:** the token-type experiment (`g`, `x`, `k`) gave no gain and was reverted (`research.md` section 14). Truncation by the top-30 rule is the real limit (misses: 1.8% over the df cap, 3.4 to 4.5% truncated; K 60 gives recall 0.9473). The fix being run is **cascade blocking**: K 100 raw candidates, then a learned first-stage ranker (`stages/prune.py`) keeps the best 30. Jobs `zc1a`, `zc1b`, `zc1c` run the chain and produce model `v2` under `runs/v2/`.
- **Code zip for the portal:** `dist/business_entity_resolution_code.zip` (local, git-ignored), all source under `src/`, README with run steps, pinned requirements. Rebuild from the final code at freeze (commands in section 9).
- A teammate's plan (`docs/archive/v2.md`, layers L0 to L5 and a harness) was reviewed; agreed integration is in section 9.

## 2. Access and identity

| Item | Value |
|---|---|
| AWS account | `567503593043` (the lead's account A, Paid plan, $200 credits) |
| Region | `us-east-1` for everything |
| CLI profile | `hxman-26`, a **root** session created with `aws login --profile hxman-26`; it expires, then re-run that command in a terminal (browser sign-in). Verify: `aws sts get-caller-identity --profile hxman-26` |
| Laptop tools | `aws` CLI v2, `session-manager-plugin` 1.2.835.0 (installed, only needed for the optional SSH-helper path), `uv`, Python 3.12 |
| Local dev env | `/home/hxman/amazon-ml-2026/.venv-ber/` (Python 3.12; includes polars, duckdb, xgboost, mlflow, anyascii, pytest). Create with `uv venv --python 3.12 .venv-ber && VIRTUAL_ENV=.venv-ber uv pip install -r code/business_entity_resolution/requirements.txt` |
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
| `code/business_entity_resolution/` | package `ber` (`src/ber`), `Makefile`, `configs/params.yaml`, `src/tests/`, `src/scripts/`, `requirements.txt`, `README.md` |
| `aws/notebook/` | `onstart.sh`, `bootstrap.sh`, `jobrunner.sh` (the infra scripts, also copied to S3 `ber/`) |
| `iam/hxman/` | IAM policy documents for the role |
| `context.md`, `research.md`, `handoff.md` | project documents |
| `docs/archive/remote-setup.md`, `remote-ssh.remote.ipynb`, `iam/*.json` | the SageMaker SSH-helper path from a teammate (optional, superseded by the job queue) |
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
aws s3 sync s3://sagemaker-us-east-1-567503593043/ber/code /home/ec2-user/SageMaker/ber --delete --exclude 'work/*' --only-show-errors --exact-timestamps
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
- `aws s3 sync` skips a file when its size is unchanged and the timestamps look compatible, so a fix that keeps the file size (for example moving a line) is NOT delivered to the notebook and the job runs the old code. Keep `--exact-timestamps` in the job header (added 26 Sep after a syntax-error fix was silently not delivered).
- A job that hangs never finishes, so a watcher that only waits for the done log never fires: also watch the age of `jobs*/live/*.log` (a 50-minute hang on 26 Sep 03:10 went unnoticed for that reason).
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
| pairs | `python -m ber.stages.pairs --split train` (7.48M pairs) / `--split test` (51.9M) | `features/{split}/part_*.parquet` (66 columns, 63 model features) | v0: about 13 s per 1.5M-pair chunk; v1: about 60 s per chunk, so test features take about 35 min (Python loops for digit strings and skeletons are the next thing to vectorise) |
| train_gpu | `python -m ber.stages.train_gpu --name v1` | XGBoost (CUDA) 5-fold grouped OOF, exclusive assignment and threshold tuning; `models/<name>/{xgb.json,config.json,report.json,oof.parquet}` | about 22 s per fold, 3 min in total |
| predict | `python -m ber.stages.predict --name v1` | `output/<name>/{matching_results.tsv,candidate_pairs.tsv}` and validation (official validator if found at `work/official/validate_submission.py`); scores part by part to fit RAM | about 2 min |
| prune (cascade, optional) | `python -m ber.stages.block --split test --k 100 --out-name test_raw`, same for `train --all-train --out-name train_raw`, then `python -m ber.stages.prune --train` and `--apply train test` | `blocks/{split}_raw/` (K 100), `models/prune/`, then `blocks/{split}/` (best 30 per S1, plus `p_block`, dropped before the matcher) | raw blocking about the same time as K 30; pruning seconds per shard |
| result checker | `python src/scripts/check_submission.py <output_dir> <test_dir>` | rule check of both TSVs (format bytes, ids exist, matches subset of candidates, one owner per record) and per-country match statistics | about 1 min |
| error analysis | `python src/scripts/error_analysis.py v1` | loss decomposition (blocking vs matcher), segments, false-positive/negative taxonomy with raw-text examples | about 2 min |

Token types in blocking: `n` name word, `a` address word, `p` 5-char prefix, `c` name x address word, `m` name-word pair, `d` address-word pair, `h` house-number x address word, and (new, being measured) `g` glued whole name, `x` one-deletion variants of the two rarest name words, `k` consonant skeleton of name words (pool side only for non-Latin names, via romanisation). The pool-index cache key includes `INDEX_VERSION` in `block.py`; bump it when token generation changes. Key parameters: `k` 30, `cap_df` 800 (tokens more frequent than this in the 10.3M pool are ignored; raising it changes nothing but costs up to 100x time), `per_type` rarest-token limits. Candidate id convention: `pid = src * 10_000_000 + rid` (`src` 2 or 3). S1 identifier is `rid`, the row index in the Parquet (0-based, equals row number in the TSV).

Local tests: `make test` or `pytest -q tests` in `code/business_entity_resolution` (16 tests including an end-to-end synthetic run of the whole v0 chain; xgboost falls back to CPU when no GPU).

## 6. What is running or pending right now

(Superseded by section 0 for the 25 Sep evening state; the table below records the earlier cascade chain.)

As of 25 Sep about 14:30 IST: **the best file is the stacked model `s1`** (`runs/s1/output/matching_results.tsv`; locked-holdout macro F0.5 0.9617, paired gain over v2 +0.0051 with 95% CI [+0.0048, +0.0055]). Calibration plus expected-F0.5 selection (`stages/expf.py`, model `e1`) was implemented exactly and gave no gain (+0.0001, CI includes 0); do not upload `e1`. Earlier this session: `zc1a`, `zc1b`, `zc1c` (cascade chain, model v2) and `zd1` (holdout scoring) are done. `zs1a` and `zs1b` (consensus stacking: build train/test consensus features, train the stacked model `s1`, paired comparison against v2 on the locked holdout, predict, publish to `runs/s1/`) are queued or running. The older rows below are kept for history.

| Job | State | Notes |
|---|---|---|
| `v0a` to `v0f`, `v1a`, `v1b`, `v1c-blockeval` | done | v0 and v1 chains; v1 outputs published at `runs/v1/`; token-type experiment (see `research.md` section 14) |
| `chk1` | run right after `v1b` | rule checker and official validator with `--check-ids` on v1 and v0 outputs (result in `jobs/done/chk1.log`) |
| `zc1a` | queued | reclaim disk (deletes the two DuckDB index files, about 35 GB), rebuild the token index for the reverted token set, block test and all train S1 with K 100 into `blocks/{split}_raw` |
| `zc1b` | queued | `prune --train`, `prune --apply train test` (writes K 30 shards to `blocks/{split}`), pair features for the train sample (vectorised, expected faster), `train_gpu --name v2`, error analysis of `v2` |
| `zc1c` | queued | test features, `predict --name v2`, rule checker, publish to `s3://sagemaker-us-east-1-567503593043/runs/v2/` |

Job names starting with `zc` sort after everything else, so other jobs run first. Jobs run one at a time in alphabetical order. After `zc1c`: the v2 file is at `runs/v2/output/matching_results.tsv`; upload only if its out-of-fold macro F0.5 beats v1 (0.9551) and the checker and validator pass.

AWS sessions expire: when a command prints "Your session has expired", the human runs `aws login --profile hxman-26` again.

## 7. Results and facts to trust (do not re-derive)

- Data structure and noise: `context.md` section 2b (row counts, singleton rate 5.6%, mean 3.46 matches per S1, exclusive ownership, France differences, noise catalogue).
- Blocking, 20k train S1 vs the full 10.3M pool, cap 800, K 30, all token types: **pair recall 0.9416** (US 0.970, India 0.899), 29.9 candidates per S1, S1 with all matches found 0.8375. Misses: 2.2% over the df cap, 3.7% lost to per-type limits and top-K. Diagnosis with no cap: only 0.01% of true pairs share no token; the rarest shared token has df at most 100 for 96%, at most 800 for 97.9%. So lexical blocking is enough; a dense channel is only for the non-Latin India minority (about 9%).
- Higher `cap_df` does not help (0.9403 at 5000, 0.9412 at 20000) and is very slow (67 s and 797 s vs 5 s).
- v1 out-of-fold (250k train S1): macro F0.5 0.9551, precision 0.989, recall 0.906; oracle on candidates 0.9781 (blocking loss 0.0219, matcher loss 0.0230). By segment: US 0.968, India 0.935, singletons 0.959, one-match entities 0.883, non-Latin match 0.897, empty-address match 0.923. F0.5 versus threshold is flat (0.9532 at 0.50, 0.9550 at 0.63, 0.9549 at 0.70).
- v0 error taxonomy and the v1 improvements are in `research.md` sections 12 and 13.
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

11. `polars.write_csv` quotes empty strings (`""`); the official validator reads that as an ID without prefix. Always `quote_style="never"` for the outputs. The old local validator hid this because pandas un-quotes; the local validator now parses raw lines.
12. The pruner's own score must not be a matcher feature (it was fit on the training S1's labels); `pairs.py` drops `p_block`.
13. The DuckDB pool-index files under `work/blocks/` are large (train about 15 GB, more with extra token types); delete them to reclaim space (they rebuild from the Parquet), and check `df -h` before adding token types.
14. Job names starting with `zc` sort last; use such prefixes to control the order of chained jobs.

## 9. Design summary and next steps

Original design is `docs/archive/plan.md` (archived) (multi-channel blocking, feature matcher, calibration, exclusive assignment, expected-F0.5 per-S1 decision, Makefile + MLflow, single account, baseline first). A teammate proposed a **record-centric** v1 (each S2/S3 record picks its owner or none; sibling consensus; fine-tuned multilingual bi-encoder; Modal); review conclusions: adopt the record-level decision with a none class and calibrated owner probability, the forensics of noise operators from matched train pairs, the evaluation discipline (locked holdout, bootstrap CI, US to India transfer as France proxy, adversarial train-vs-test validation, an empty submission to measure the test singleton share), and sibling features as second-stage stacking; postpone dense-first retrieval, FAISS-GPU, Modal and the fine-tuned e5 until v0 shows where India is weak. Do not rebuild `prepare`, `block` or `features`: extend them.

Suggested order from here (evidence in `research.md`, sections 12 to 15):
1. **Finish the cascade run** (`zc1a` to `zc1c`): check the pruner report first (pair recall of the best 30 versus the top 30 by blocking score; target above 0.9416, K 60 reaches 0.9473). If recall improves and out-of-fold F0.5 beats v1, `v2` replaces v1.
2. **Scores for all train S1.** The pair model is trained on 250k sampled S1, but competition in a later record-level layer needs p1 for every S1 (test has all of them). S1 outside the sample were never trained on, so scoring them with the final model gives unbiased probabilities. Export `p1` for all train S1 and for test (`predict` currently discards test p1; save it as `pair_p.parquet`).
3. **Locked holdout.** Draw a disjoint holdout of about 150k S1 from the remaining 1.95M train S1 (blocking already covers all S1; only pair features are needed), plus bootstrap confidence intervals, so gains are provably real. US-to-India transfer is the France proxy; add a per-country drift monitor on test predictions.
4. **Stage-2 stacking with consensus features** (do the other records of the same S1 agree on digits and name variants; margin of the record to its best other S1): targets the look-alike distractors, 84% of the false positives. Teammate's layer L2c.
5. **Calibration and per-S1 expected-F0.5 selection** (exact algorithms exist, see `research.md`), then the owner layer L3 only if it helps (exclusive assignment already gives the same score because `margin_p` and `rank_p` encode competition).
6. Cross-encoder on the ambiguous band (mmBERT-small, MIT, about 140M) only if steps 4 and 5 leave a gap; it occupies the single GPU and queue for about an hour of fine-tuning.
7. Romanisation dictionary, dense channel: last; non-Latin is now 0.897 against an oracle of 0.920 (at most about 0.25 overall).
8. Freeze, reproduce from scratch, methodology write-up from `docs/Documentation_template.md` in S3, rebuild the code zip, package layout in `context.md` section 1.

Integration with the teammate's v2 plan (agreed in review): they own the evaluation harness, L2c, L3/L4, cross-encoder and dictionary work, on a branch that adds new modules only; the pipeline owner keeps `stages/` (prepare, blocking, features, pair model, cascade) and delivers the data contracts: candidates `blocks/{split}/cand_*.parquet` (`q` S1 row id, `pid = src * 10_000_000 + rid`), `models/<name>/oof.parquet` (`q, pid, p, label`), test `pair_p.parquet`, features `features/{split}/part_*.parquet`. Their access to AWS must be a scoped IAM user (working bucket prefixes `jobs/`, `runs/`, `ber/` plus read on the data bucket, keys created by the human outside chat), never the root login; or the pipeline owner runs their jobs.

Rebuild the code zip from the repo root:
```bash
cd code && python3 - <<'PY'
import zipfile, os
root = "business_entity_resolution"; skip = {"__pycache__", ".pytest_cache", "work", "models", ".venv", "mlruns"}
with zipfile.ZipFile("../dist/business_entity_resolution_code.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for d, dirs, files in os.walk(root):
        dirs[:] = sorted(x for x in dirs if x not in skip)
        for f in sorted(files):
            if not f.endswith((".pyc", ".tmp")): z.write(os.path.join(d, f))
PY
```
then unzip into an empty folder, create a venv from `requirements.txt` and run `pytest -q src/tests` before uploading.

- Team **Nooglers**: Roshan T (team leader), Hariheman V K, Sai Nivedh V, Baranidharan Selvaraj. Final package `Nooglers_submission.zip`. Contact details are deliberately not stored in the repo.

## 10. Rules

- No secrets, keys, presigned URLs or login links in git or chat. `aws login` is run by the human in their own terminal.
- No external data lookups (entity-resolution APIs, registries, geocoding). Pretrained models must be MIT or Apache-2.0 and at most 8B parameters; log licences.
- Portal uploads and any action with side effects outside the account (submissions, billing changes, deleting data) are done or approved by the human.
- Stop the notebook when idle for long periods.
- Log every experiment (`runs.jsonl` plus MLflow) with its validation number so the submission history required by the guidelines exists.
- Commit documentation and code changes to branch `sai`; run tests before syncing to S3.
- Never share the root AWS login; give collaborators a scoped IAM user and let the human create the keys outside the chat.
