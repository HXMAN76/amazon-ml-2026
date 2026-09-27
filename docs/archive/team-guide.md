# Team guide: how we train, what the best model is, what is left to do

> **Continuing on your own AWS? Start with `TEAMMATE_HANDOFF.md`** (state at 26 Sep 22:45 IST, what to submit next, where every artifact is, how to run things without our account).

Audience: the Nooglers teammates (Roshan T, Sai Nivedh V, Baranidharan Selvaraj) and any agent working for them. Last updated 26 Sep 2026 about 16:30 IST (best: `s17`, holdout 0.99025, portal 0.981). What runs on every GPU and the ideas still open: `EXPERIMENTS.md`. Full version history and component reference: `ARCHITECTURE.md`. Details behind every number are in `handoff.md` (infra, section 0 for the latest state), `research.md` (measurements, sections 12 to 19), `context.md` (data facts). Window closes **Sun 27 Sep 2026 23:59 IST**; 5 portal submissions per day; one login at a time.

## 1. Where we are

| File | Portal (public subset) | Locked holdout F0.5 | Candidates per S1 | Where |
|---|---|---|---|---|
| `v2` first pair model | 0.944 | 0.9565 | 30 | `runs/v2/output/` |
| `s3all` stack | 0.949 | 0.9677 | 30 | `runs/s3all/output/` |
| `s4` stack with name dense channel | 0.953 | 0.9708 | 32 | `runs/s4/output/` |
| `s5` stack with name+address dense channel | not yet scored | **0.9832** | 31 to 36 | `runs/s5/output/` |
| `s6` = `s5` with candidate shortlist | not yet scored | 0.9831 | **4.9** | `runs/s6/output/` |
| `s8` decoy edit + extra columns | not yet scored | 0.98362 | 4.9 | `runs/s8/output/` |
| `s12` + 1.5M S1, depth 9 | not yet scored | 0.98505 | 4.9 | `runs/s12/output/` |
| `s11` + cross-encoder score | not yet scored | 0.98887 | 4.9 | `runs/s11/output/` |
| `s13` `s11` + 1.5M S1 + depth 9 | not yet scored | 0.98917 | 4.9 | `runs/s13/output/` |
| `s15` stack on the first stage `v7` (850k S1) with cross-encoder 1 | not yet scored | 0.98935 | 4.74 | `runs/s15/output/` |
| `s14` base `v5`, cross-encoder 2 (e5-base) | not yet scored | 0.98986 | 4.9 | `runs/s14/output/` |
| **`s17`** base `v7`, e5-base and e5-small cross-encoders, refined competition | **0.981** (26 Sep 12:31) | **0.99025** | 4.74 | `runs/s17/output/` |
| `s12` (uploaded 26 Sep) | **0.971976 (rank 402)** | 0.98505 | 4.9 | `runs/s12/output/` |

**`s22` is the best by holdout (0.99054); `s17` is the best on the portal so far (0.981).** **The portal gap is France:** country probes gave France only 0.187 and US only 0.453, i.e. France F0.5 about 0.92 and US plus India about 0.99 (`research.md` sections 23 and 24). Next portal candidates: `s22f985`, `s22f97`, `s22f995` (`s22` with only France's threshold changed). The organisers rank `candidate_pairs.tsv` too and reward a smaller candidate set; all files above have 4.7 to 4.9 candidates per S1. All paths are under `s3://sagemaker-us-east-1-567503593043/`. Holdout = 150k train S1 that no model trains on (seed 2026, `ber.split.holdout_q`); every claim is a paired bootstrap on it. The portal has sat 0.009 to 0.018 below the holdout (adversarial validation AUC 0.87: the test has fewer S1, about 40% unowned pool records, 47% India, 15% France with no labels).

**In flight (26 Sep 13:45):** `s18` small cross-encoder, `v8`/`s19` joint name counts, `v9`/`s21` shift-robust first stage, `s20` e5-large cross-encoder, `s22` cross-encoder on every shortlisted pair, `v10`/`s23` test-like universe. Live list and owners: `EXPERIMENTS.md` sections 6 and 7.

## 2. Architecture (stages up to `s6`; the `s17` additions are listed after the diagram)

Task: each Source 1 (S1) record has 0 to 11 matches among S2/S3 records; each S2/S3 record has at most one owner. Metric: macro F0.5 per S1 (an S1 with no match scores 1 only for an empty prediction).

```
raw TSV -> prepare (normalise text; France rules; romanise non-Latin)
 -> candidates = union of
      token blocking   DuckDB IDF-weighted token index, 100 per S1, learned pruner keeps the best 30
      dense (names)    multilingual-e5-small fine-tuned; non-Latin pool names -> nearest S1 (top 5)      [v3]
      dense_all        e5-small over "name | address" of EVERY record, fine-tuned with hard negatives;
                       pool record -> nearest S1 (top 1; top 3 if the pool address is empty)          [v5+]
 -> first stage        67 pair features (string similarity, name rarity, competition margin_p / rank_p,
                       house number, romanised similarity, dense cosines) -> XGBoost (GPU) -> p1
 -> shortlist          per S1 the best 10 by p1 with p1 >= 0.005 (always the best one)        [s6]
 -> stack (stage two)  XGBoost on p1-based consensus: how many confident records the S1 already has, how strongly other
                       S1 claim the same record, house-number / digit agreement with the S1's other records,
                       TF-IDF cosine of names and addresses
 -> decision           exclusive assignment (a record goes to its highest-p S1), threshold about 0.67 tuned out-of-fold
 -> outputs            matching_results.tsv, candidate_pairs.tsv (exactly the shortlist the stack scored)
```

Added after `s6`: two cross-encoders (`multilingual-e5-base`, `-small`) score the uncertain band of p1 and give the stack `xs`, `xs2`; first stage on 850k S1; stack on 1.5M S1 with decoy edit features, carried columns and competition on the refined probability. Full description: `ARCHITECTURE.md` section 3.0.

What each piece bought (holdout): cascade blocking +0.001; stacking with consensus and digit features +0.011; TF-IDF +0.0005; name dense channel +0.0033 (first stage) and +0.0031 in the stack; name+address dense channel **+0.016 first stage, +0.012 stack** (blocking loss 0.0166 to 0.0047, India 0.957 to 0.981); shortlist: about 84% fewer candidates at no score cost. Tried and dropped: calibration plus expected-F0.5 selection (null), extra token types, top-3 and top-5 for the name+address channel (18 to 38M extra pairs for almost nothing).

Code (all in `code/business_entity_resolution/src/ber/`): `text.py` normaliser, `stages/prepare.py`, `block.py`, `prune.py`, `dense.py` (names), `dense_all.py` (name+address), `pairs.py` (features), `train_gpu.py`, `score_rest.py`, `predict.py`, `stack.py` (consensus, TF-IDF, shortlist), `decision.py`; analysis scripts in `src/scripts/` (`paired_models.py`, `noise_analysis.py`, `miss_analysis.py`, `error_analysis.py`, `country_expected.py`, `shortlist_eval.py`, `check_submission.py`); parameters in `configs/params.yaml`.

## 3. How to use the AWS setup

Everything runs on one SageMaker notebook instance, driven through an S3 job queue. No SSH is needed.

| Item | Value |
|---|---|
| Account / region | `567503593043` (Hariheman's account), `us-east-1` |
| Notebooks | `test-notebook` (`ml.g5.16xlarge`, 1 A10G, $5.12/h, lanes `jobs/` `jobs2/`) and `test-notebook-2` (`ml.g5.24xlarge`, 4 A10G, lanes `jobsB/` `jobsB2/` `jobsB3/` `jobsB4/`) |
| Bucket | `sagemaker-us-east-1-567503593043`: `ber/code` (code the jobs run), `jobs/{pending,live,done}` (queue and logs), `runs/<name>/` (published outputs and models) |
| Data | `s3://ml-challenge-nooglers/ml-challenge-2026/raw/v1/` (teammate account); already copied onto the notebook |
| On the notebook | code `/home/ec2-user/SageMaker/ber`, data `.../dataset`, all intermediate files `.../work` (about 70 GB of 98 GB; `blocks/`, `features/`, `stack/`, `models/`, `output/`, `dense_all/`) |

**Run a job**
1. Write a bash script; start with the standard header (copy from `handoff.md` section 4): sync code from S3, `cd`, export `BER_DATA`, `BER_WORK`, `PYTHONPATH`.
2. Publish the code you changed: `aws s3 sync code/business_entity_resolution s3://sagemaker-us-east-1-567503593043/ber/code --delete --exclude '*__pycache__*' --exclude 'models/*' --exclude 'work/*'`.
3. Queue it: `aws s3 cp myjob.sh s3://sagemaker-us-east-1-567503593043/jobs/pending/myjob.sh`.
4. Watch: `aws s3 cp s3://.../jobs/live/myjob.log -` while running, `.../jobs/done/myjob.log` after (last line `exit=<code>`).

**Rules that keep us from breaking each other's work**
- The queue is sequential and alphabetical: one job at a time, one GPU. A long job blocks everyone. Tell Hariheman before queueing anything over 30 minutes.
- Use unique job names and unique model names with your initials (for example `bs_xenc1`). Our runs use `v*`, `s*`, `z*`; a name that already exists in `work/models` is overwritten.
- Do not sync your branch into `ber/code` (our jobs run whatever is there). Use your own prefix (`ber/code_<initials>`) and sync from it in your job header.
- Always keep `--exclude 'work/*'` on syncs; without it `--delete` erases the data disk.
- Never put credentials, presigned URLs or Jupyter links in chat or git. Nothing under `handoff.md` contains a secret; keep it that way.
- Cost: every hour of the big notebook costs money; the notebook stops itself after an idle hour (job-aware). Do not leave experiments running without a purpose.

**Second notebook (26 Sep).** `test-notebook-2` (`ml.g5.24xlarge`, 4 GPUs) has its own queue prefixes `jobsB/` to `jobsB4/` (pin each job with `CUDA_VISIBLE_DEVICES`) (the runner picks the prefixes from the notebook name); the working files it needs are exported to `s3://sagemaker-us-east-1-567503593043/state/` by a job on the main notebook. Check heartbeats in `diag/<notebook-name>.txt`. The AWS CLI login expires after about 5 hours: `aws login --profile hxman-26`.

**Access for teammates.** The current CLI login is the account root, which must not be shared. To give a teammate queue access without the console, Hariheman creates an IAM user with the policy in `iam/team/teammate_s3_policy.json` (list/read `runs/`, `jobs/`, `ber/code*`; write only `jobs/pending/` and `ber/code_*`) and hands over the keys privately (not in chat or the repo). Note that anyone who can write to `jobs/pending/` can run code on the notebook, so treat that access as notebook access. Alternative with no new credentials: send a job script or a branch name to Hariheman, who queues it.

## 4. Optimization list

| # | Item | Status | Expected gain | Owner |
|---|---|---|---|---|
| 1 | Extra dense neighbours for empty-address pool records | done, no gain | 0 | pipeline |
| 2 | Bigger box and parallel lanes | done (two notebooks, six lanes) | speed | pipeline |
| 3 | Iterated consensus | done (`s9`), no gain | 0 | pipeline |
| 4 | Decoy edit features | done (`s8`, +0.0005) | | pipeline |
| 5 | State-name normalisation (`OK` and `Oklahoma`, Indic state names) | to do; needs re-prepare | about +0.001 | teammate or pipeline |
| 6 | Cross-encoders | done (`s11` to `s17`, +0.005 in total) | | pipeline |
| 7 | Bigger first stage and stack | done (850k and 1.5M S1) | | pipeline |
| 0 | **France** (about 0.010 of score): threshold sweep, calibration by distribution matching, France-aware features (see `EXPERIMENTS.md` section 10) | running and next | +0.001 to +0.009 | Hariheman and pipeline; teammates welcome (no France labels: only portal probes measure it) |
| 8 | Shift remedies (mostly settled: the gap is France, not distractor density): `v8` joint counts, `v9` drop blocking features, `v10` test-like universe, `s22` cross-encoder on all shortlisted pairs, `s20` e5-large | running | closes part of the 0.009 portal gap | pipeline |
| 9 | Untried: density-ratio weights, cross-encoder test-time augmentation, cross-fitted cross-encoder, learned per-S1 shortlist, cap 5 S2 and 6 S3 | open, good teammate tasks | +0.0002 to +0.001 each | anyone |
| 10 | Portal probes: `s17f85`, per-country files (`reemit.py`), France threshold sweep | next slots, more may open tomorrow | up to +0.002 | Hariheman |
| 11 | Freeze: rebuild code zip, one clean reproduction, official validator `--check-ids`, methodology document, final uploads | Sunday | none | all |

Where the loss still is (holdout, `s5`): blocking 0.0047 (80% of never-proposed pairs have an empty pool address), matcher 0.0121 (name noise: injected or dropped words, typos, alias-only names; 78% of false positives are records with no owner). Empty-address pool records: 4.4% of true pairs, 71% in candidates, 50% matched.

## 5. Rules and portal process
- Only the provided training data, no external lookups; models MIT or Apache-2.0 and under 8B parameters (ours: XGBoost, `multilingual-e5-small` 118M and `-base` 278M, MIT).
- 5 submissions per day, one login at a time, keep the version history (portal log in `submission_checklist.md` section F).
- Upload only files that beat the current best on the locked holdout by more than the paired interval; the final pick is by holdout and portal together.
- Final package: `Nooglers_submission.zip` with `output/`, `code/business_entity_resolution/{src,README.md,requirements.txt}` and the filled methodology document (draft in `submission/Documentation_template.md`).
