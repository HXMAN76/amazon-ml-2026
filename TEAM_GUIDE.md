# Team guide: how we train, what the best model is, what is left to do

Audience: the Nooglers teammates (Roshan T, Sai Nivedh V, Baranidharan Selvaraj) and any agent working for them. Last updated 26 Sep 2026 about 02:15 IST. Full version history and component reference: `ARCHITECTURE.md`. Details behind every number are in `handoff.md` (infra, section 0 for the latest state), `research.md` (measurements, sections 12 to 19), `context.md` (data facts). Window closes **Sun 27 Sep 2026 23:59 IST**; 5 portal submissions per day; one login at a time.

## 1. Where we are

| File | Portal (public subset) | Locked holdout F0.5 | Candidates per S1 | Where |
|---|---|---|---|---|
| `v2` first pair model | 0.944 | 0.9565 | 30 | `runs/v2/output/` |
| `s3all` stack | 0.949 | 0.9677 | 30 | `runs/s3all/output/` |
| `s4` stack with name dense channel | 0.953 | 0.9708 | 32 | `runs/s4/output/` |
| `s5` stack with name+address dense channel | not yet scored | **0.9832** | 31 to 36 | `runs/s5/output/` |
| `s6` = `s5` with candidate shortlist | not yet scored | 0.9831 | **4.9** | `runs/s6/output/` |

`s6` is the file to upload for the final ranking: the organisers rank `candidate_pairs.tsv` too and reward a smaller candidate set. All paths are under `s3://sagemaker-us-east-1-567503593043/`. Holdout = 150k train S1 that no model trains on (seed 2026, `ber.split.holdout_q`); every claim is a paired bootstrap on it. The portal has sat 0.012 to 0.018 below the holdout every time (test is harder: more distractors, 47% India, 15% France with no labels).

## 2. Architecture of the best model (`s5` / `s6`)

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

What each piece bought (holdout): cascade blocking +0.001; stacking with consensus and digit features +0.011; TF-IDF +0.0005; name dense channel +0.0033 (first stage) and +0.0031 in the stack; name+address dense channel **+0.016 first stage, +0.012 stack** (blocking loss 0.0166 to 0.0047, India 0.957 to 0.981); shortlist: about 84% fewer candidates at no score cost. Tried and dropped: calibration plus expected-F0.5 selection (null), extra token types, top-3 and top-5 for the name+address channel (18 to 38M extra pairs for almost nothing).

Code (all in `code/business_entity_resolution/src/ber/`): `text.py` normaliser, `stages/prepare.py`, `block.py`, `prune.py`, `dense.py` (names), `dense_all.py` (name+address), `pairs.py` (features), `train_gpu.py`, `score_rest.py`, `predict.py`, `stack.py` (consensus, TF-IDF, shortlist), `decision.py`; analysis scripts in `src/scripts/` (`paired_models.py`, `noise_analysis.py`, `miss_analysis.py`, `error_analysis.py`, `country_expected.py`, `shortlist_eval.py`, `check_submission.py`); parameters in `configs/params.yaml`.

## 3. How to use the AWS setup

Everything runs on one SageMaker notebook instance, driven through an S3 job queue. No SSH is needed.

| Item | Value |
|---|---|
| Account / region | `567503593043` (Hariheman's account), `us-east-1` |
| Notebook | `test-notebook`; now `ml.g5.xlarge` (1 A10G, 4 vCPU); moving to `ml.g5.16xlarge` (64 vCPU, 256 GB, 1 A10G, $5.12/h) for the last two days |
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

**Access for teammates.** The current CLI login is the account root, which must not be shared. To give a teammate queue access without the console, Hariheman creates an IAM user with the policy in `iam/team/teammate_s3_policy.json` (list/read `runs/`, `jobs/`, `ber/code*`; write only `jobs/pending/` and `ber/code_*`) and hands over the keys privately (not in chat or the repo). Note that anyone who can write to `jobs/pending/` can run code on the notebook, so treat that access as notebook access. Alternative with no new credentials: send a job script or a branch name to Hariheman, who queues it.

## 4. Optimization list

| # | Item | Status | Expected gain | Owner |
|---|---|---|---|---|
| 1 | Extra dense neighbours for name-only pool records (empty address) | done, no gain (`v6` 0.97557, `s7` 0.98309, same as `s6`); dropped | 0 | pipeline |
| 2 | Bigger box (16xlarge) and parallel job lanes | after `s7`, about 02:30 | speed only | pipeline |
| 3 | Iterated consensus (stack on the stack's probabilities) | to do | +0.001 to +0.003 | pipeline |
| 4 | Decoy edit features (substitution vs indel, Hamming distance, length difference) | to do | about +0.001 | pipeline or teammate |
| 5 | State-name normalisation (`OK` and `Oklahoma`, Indic state names) | to do; needs re-prepare and rebuild | about +0.001 | pipeline or teammate |
| 6 | Cross-encoder on the uncertain band (p1 between 0.05 and 0.95), one extra stack feature | teammate's own work; we can add its score as a feature | +0.002 to +0.004 | teammate |
| 7 | Seed ensemble of the stack, first stage and stack on 2 to 3 times more training S1 | to do | +0.001 to +0.002 | pipeline |
| 8 | Learned per-S1 shortlist size; check floor 0.02 | at the freeze | smaller file at about -0.0001 | pipeline |
| 9 | Cap of 5 S2 + 6 S3 matches per S1 | at the freeze | up to +0.0002 | pipeline |
| 10 | Threshold check on the portal (test has more distractors) | Saturday evening, 2 to 3 uploads | up to +0.002 | Hariheman |
| 11 | Freeze: rebuild code zip, one clean reproduction, official validator `--check-ids` on the notebook, methodology document, final uploads | Sunday | none | all |

Where the loss still is (holdout, `s5`): blocking 0.0047 (80% of never-proposed pairs have an empty pool address), matcher 0.0121 (name noise: injected or dropped words, typos, alias-only names; 78% of false positives are records with no owner). Empty-address pool records: 4.4% of true pairs, 71% in candidates, 50% matched.

## 5. Rules and portal process
- Only the provided training data, no external lookups; models MIT or Apache-2.0 and under 8B parameters (ours: XGBoost and `multilingual-e5-small`, 118M, MIT).
- 5 submissions per day, one login at a time, keep the version history (portal log in `submission_checklist.md` section F).
- Upload only files that beat the current best on the locked holdout by more than the paired interval; the final pick is by holdout and portal together.
- Final package: `Nooglers_submission.zip` with `output/`, `code/business_entity_resolution/{src,README.md,requirements.txt}` and the filled methodology document (draft in `submission/Documentation_template.md`).
