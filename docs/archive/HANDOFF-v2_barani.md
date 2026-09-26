> **ARCHIVED (26 Sep 2026): A teammate's handoff of 25 Sep; the current handoff is handoff.md. Kept for history only; do not follow its instructions.**

# Project Handoff

Written 2026-09-25 ~05:20 IST by a Claude Code session on the laptop of the teammate (user `barani200`, git user `imbaraniii`). Companion to the lead's `handoff.md` (v0 infrastructure, tracked on `origin/sai`); read both.

## 1. Session Overview
- **Larger objective.** Amazon ML Challenge 2026 on Unstop, "Business Entity Resolution". For every Source 1 (S1) business, list the matching Source 2/3 (pool) records. The metric is macro F0.5 per S1 (singletons score 1.0 only for an empty prediction). The window closes **Sun 27 Sep 2026 23:59 IST**; 5 submissions per day. Team of 4; the lead is HXMAN76, who owns the v0 pipeline.
- **Starting state.**
  - The repo held a generic pre-task pipeline on `main`.
  - The branch `sai` held the lead's early v0 (`5231dc4`) plus SageMaker SSH notes.
  - This laptop has no data and no account-A AWS access.
- **Work done, in order:**
  1. Explained the problem statement (`ml-challenge.pdf`, `prob_statement.pdf`).
  2. Wrote `understanding.md`.
  3. Built a first standalone pipeline `er` (stashed; obsolete).
  4. Designed and **built v1 record-centric** on branch `v1/record-centric` (tested on synthetic data only).
  5. The lead reviewed v1: good ideas, stale premises (v0 already ran at scale).
  6. Deep web research → `research-v2.md`.
  7. Redesigned as **v2**, then **v2.1** after a second lead review → the plan of record is `v2.md`.
  8. Created branch `v2/decision-layer` from `origin/sai`.
  9. Implemented **step 1: the evaluation harness** `stages/v2_eval.py` + `tests/test_v2.py` (passing).
- **Where it stopped.**
  - The harness is written and tested but **not committed**.
  - I asked "commit to `v2/decision-layer`, then start the stage-2 stacking model?" and got no answer.
  - The user then switched the working tree to local `sai`, which is old (7 behind `origin/sai`). The v2 files are untracked, so they came along to `sai`, but they don't work there.
  - The user then ran `/checkpoint`.

## 2. Current Objective
- **Goal.** Build the v2.1 layers on top of the lead's v0 so the holdout macro F0.5 beats v0 by more than the bootstrap CI.
- **Acceptance per layer:** the paired-bootstrap delta vs the previous best run has a 95% CI lower bound > 0 (`v2_eval --baseline` prints `ship=True`).
- **Constraints:**
  - Only the provided data, and no external lookups (disqualification risk).
  - Pretrained models only MIT/Apache-2.0, ≤ 8B parameters; log licences.
  - Outputs are `matching_results.tsv` + `candidate_pairs.tsv`, TSV, validated by the official `utils/validate_submission.py`.
  - Final zip: `output/` + `code/business_entity_resolution/{src,README.md,requirements.txt}` + filled `Documentation_template.md`.
  - The version history of submissions must be kept.

## 3. Project / System Context
- **Data (never download to the laptop; slow home network).** `s3://ml-challenge-nooglers/ml-challenge-2026/raw/v1/` (`dataset/{train,test}/*.tsv` ≈ 2.9 GB, `utils/validate_submission.py`, `docs/Documentation_template.md`).
  - Train: 2.2M S1 and ~10M pool records.
  - Test: 1.73M S1; France appears only in test.
  - The S1 singleton rate is 5.6%; mean 3.46 matches per S1; ownership is exclusive (no pool record claimed by 2 S1s).
- **Compute:** SageMaker notebook `test-notebook` (ml.g5.xlarge, A10G 24 GB, 4 vCPU, ~15 GB RAM) in the lead's AWS account A (`567503593043`, us-east-1).
  - It is driven by an **S3 job queue**: a `.sh` file put in `s3://sagemaker-us-east-1-567503593043/jobs/pending/` runs; the log appears in `jobs/done/`.
  - Details: the lead's `handoff.md` §2–4.
- **v0 pipeline** (the lead's, `code/business_entity_resolution/`, package `ber`, stages in `src/ber/stages/`):
  - `prepare` (TSV → Parquet + `labels.parquet`)
  - `sample` (250k train S1 + 5 folds)
  - `block` (DuckDB token-index blocking, top-30 per S1)
  - `pairs` (vectorised features)
  - `train_gpu` (XGBoost, grouped OOF, threshold, exclusive option)
  - `predict`
  - Knobs are in `configs/params.yaml`; `Makefile` target `reproduce`; tracking via `ber.tracking.log_stage` (runs.jsonl + optional MLflow).
- **v0 ids (do not rename):**
  - `q` = S1 rid.
  - `pid` = src × 10,000,000 + pool rid (S2 = 2, S3 = 3).
  - `BER_WORK` = work dir; `BER_DATA` = raw TSV dir.
- **v0 files v2 reads:**
  - `WORK/sample/train_s1.parquet` (`rid, ctry, n_matches, fold`)
  - `WORK/models/<run>/{config.json (threshold, exclusive, features), oof.parquet (q, pid, p, label)}`
  - `WORK/parquet/train/{source1,2,3,labels}.parquet` (source columns include `addr`, `nl_name`, `ctry`, `core1`, `name2`, `legal`)
- **Files the lead agreed or was asked to add:**
  - `WORK/sample/holdout_s1.parquet` (~150k disjoint S1)
  - `WORK/models/<run>/pair_p_{train_all,holdout}.parquet`
  - `WORK/output/<run>/pair_p.parquet` (test p; today it is discarded after predict)

## 4. Current Implementation State
- **Latest v0 figures (from the lead's second review, as reported in chat):**
  - OOF macro F0.5 **0.9551** (India 0.935, non-Latin 0.897 vs oracle 0.920, singletons 0.959)
  - loss 4.5 points = blocking recall 2.19 + matcher 2.30
  - exclusive assignment = plain (0.9551)
  - flat threshold curve (0.9532 at 0.50, 0.9550 at 0.63)
  - 84% of FPs are look-alike distractors
  
  Earlier figures in `research.md` §12 on `origin/sai` (0.9377) are from before the lead's feature pass.
- **v2 code (untracked, in the working tree):**
  - `code/business_entity_resolution/src/ber/stages/v2_eval.py` (227 lines), the evaluation harness:
    - `entity_f05` (per-S1 F0.5, same rule as `decision.macro_f05`)
    - `bootstrap_ci`, `v0_select` (exclusive + threshold)
    - `truth_pairs`, `with_label`
    - `segment_table` (country, match-count bucket, non-Latin match, empty-address match, shared address)
    - `score_split` (macro + CI + oracle-on-candidates loss split + breakdowns)
    - `paired_delta` (paired bootstrap vs a baseline run; `ship` flag)
    - `drift_report` (test vs dev per country; alert if the share-with-match differs by > 0.05 or the country is unseen)
    - `main` CLI: `--run`, `--selected DIR` (`selected_{dev,holdout,test}.parquet`), `--baseline RUN`, `--tag`, `--boot`
    - Writes `WORK/v2/<tag>/eval.json` and `entity_{split}.parquet`.
    - The holdout section runs only if `holdout_s1.parquet` + `pair_p_holdout.parquet` exist; drift only if `output/<run>/pair_p.parquet` exists.
    - `PID_BASE` is defined locally (not imported from `stages.block`, which imports DuckDB at module level).
  - `code/business_entity_resolution/tests/test_v2.py` (117 lines): 7 tests (metric agreement with v0, hand-checked loss split, paired ship gate, exclusive selection, drift alert, a full CLI run on a fake WORK in v0 layout, and a `PID_BASE` equality check that needs DuckDB).
- **Works:** all harness tests pass locally (on `v2/decision-layer`).
- **Incomplete:**
  - Nothing is committed.
  - Not yet run on real data.
  - Steps 2–6 of the build order are not started.
- **Broken:** on local `sai` (current HEAD) the v2 files cannot run: old `sai` lacks v0's `decision.py`, `tracking.py`, `config.py` and `stages/`.

## 5. Detailed Session Changes (chronological)
1. **`understanding.md` (untracked, repo root).** A full write-up of the problem, metric, dataset and v0 as known then. Estimates are tagged; the dataset numbers came from `plan.md` / header peeks. Some figures are now superseded (see §4).
2. **Standalone `er` pipeline** (`business_entity_resolution/src/er/`, on the local branch `er`, then stashed as `stash@{0}` "er: entity-resolution pipeline WIP"). **Obsolete**; duplicates the lead's work.
3. **v1 record-centric (branch `v1/record-centric`, never committed).**
   - New modules: `canon.py` (Indic romaniser), `prepare.py`, `embed.py`, `retrieve.py`, `evaluate.py`, `forensics.py`, `tests/test_v1.py`.
   - Rewrites of tracked files: `run.py`, `features.py`, `model.py`, `decide.py`, `synth.py`, `data.py`, `README.md`, `requirements.txt`, `aws/notebook/bootstrap.sh`.
   - Plus job scripts `aws/notebook/jobs/{_common.sh,v1_*.sh}`.
   - 6 tests passed on synthetic data.
   - **The tracked-file rewrites were later reverted in the working tree** (apparently a `git restore`/checkout by the user). The untracked new modules survived and were moved to `archive/v1/`.
4. **`v1.md`** (untracked): the v1 design + implementation notes. **Superseded.**
5. **`research-v2.md`** (untracked): research on models and approaches with sources: Foursquare 2022 Kaggle solutions, NIL prediction, GraLMatch/TransClean, IndicXlit, MuRIL, gte reranker, Qwen3, LinkTransformer, Shiprocket address NER, jellyfish.
6. **`v2.md`** (untracked): **plan of record v2.1** (architecture, priorities by measured loss, data contracts in v0 naming, leakage rules, evaluation harness, operations, build order, files, verification, open items).
7. **Branch `v2/decision-layer`** created from `origin/sai` (fd725b2).
   - Before creating it, the untracked local `plan.md` (an older copy of the lead's plan) collided with the tracked one and was renamed `plan.local-old.md`.
   - The v1 leftovers were moved to `archive/v1/` so they don't break v0's pytest or get synced to S3.
8. **`stages/v2_eval.py` + `tests/test_v2.py` written.**
   - Fixed a drift-report bug: `ctry` was duplicated inside each entry because the dict value was evaluated before `r.pop` in the key expression.
   - Tests: `tests/test_v2.py tests/test_text.py` → 16 passed, 1 skipped (the DuckDB-dependent `PID_BASE` check).
9. **The user switched to local `sai`** (HEAD now `sai` @ 5231dc4, behind `origin/sai` by 7).

## 6. Technical Decisions and Reasoning
- **v0 is the backbone** (the lead's code; do not rebuild `prepare`/`block`/`pairs`/`train_gpu`). v2 adds only new modules on `v2/decision-layer`.
- **Final architecture (v2.1, in `v2.md`):**
  1. Stage 1: v0 blocking, plus the lead's upgrade.
  2. Stage 2a: v0 XGBoost p1, scored for all train S1 (the density fix).
  3. Stage 2b: cross-encoder (mmBERT-small, MIT) on the uncertain band only, gated at ≥ +0.005 and ≤ 60 min inference.
  4. Stage 2c: stage-2 XGBoost with consensus / sibling-agreement and record-side competition features → p2.
  5. Stage 3: isotonic calibration → optional record owner-or-none → per-S1 P(any) + expected-F0.5 prefix + postcode/house vetoes.
  6. Stage 4: outputs + official validator.
  7. Across all stages: the harness (locked holdout, CI, loss split, US↔India transfer, adversarial check, France drift monitor, LB log).
- **Order by measured loss (the lead's correction):**
  1. blocking upgrade (lead)
  2. stage-2 consensus stacking
  3. decision + calibration
  4. cross-encoder
  5. owner-or-none (only if it helps)
  6. India dictionary / dense retrieval (last; ≤ ~0.25 point)
- **Record-centric retrieval was dropped:** it doubles the pairs (~100M vs 52M) for no recall gain.
- **The record-centric decision was demoted to optional:** v0's `rank_p` / `margin_p` (its top features) already encode "another S1 is a better owner", and exclusive assignment gave the identical score.
- **Density fix (the lead found it):** train p1 existed only for the 250k sample, so competition features would be understated in train compared with test. Fix: score all ~66M train pairs; non-sample S1 were never trained on, so their p1 is unbiased.
- **Locked holdout:** ~150k S1 disjoint from the 250k dev sample, never trained on.
- **Rejected / postponed:**
  - Modal (card, credentials handed to a third party; the GPU isn't the bottleneck)
  - FAISS-GPU (wheel risk; torch matmul top-k instead)
  - pool-to-pool kNN over 10M records (too heavy; consensus features instead)
  - fine-tuned e5 / dense retrieval as the backbone
  - libpostal (OSM-derived knowledge; needs an organiser ruling)
  - Llama/Gemma-based models (licence)
  - Unicorn and `lt-wikidata-comp-multi` (no stated model licence)
  - an LLM adjudicator
- **Realistic headroom: ~1.5–2.5 points** (blocking ~1–1.5, stacking ~0.5–1, decision ~0.3).
- **Access:** the root login is never shared. Options: a scoped IAM user for this laptop (recommended) or the lead runs our jobs. **Undecided.**

## 7. Failed Approaches
- **v1 as a replacement for v0.** Premise wrong (v0 already scaled; recall 0.942); superseded. Its code is kept in `archive/v1/` for porting (`canon.romanize`, `forensics._align/_subs_map`, `evaluate.py` bootstrap). `decide.py`'s `expected_f05` DP must be re-created: its tracked-file version was reverted. The DP was verified against brute force in `archive/v1/test_v1.py::_brute`.
- **v1 evaluation-protocol bug (fixed then; lesson for v2).** Assigning "records touching a hold S1" to the holdout stole dev S1s' own records (dev 0.63 vs hold 1.00). Correct rule: records follow their owner's group; distractors are hashed; every prediction comes from a model that never saw that record.
- **Unnormalised token IDF** made train/test scales differ (log N_S1 differs). Normalise by log N.
- **macOS OpenMP clash:** importing torch before xgboost, or torch after sklearn+xgboost have loaded, **segfaults**. Keep torch out of CPU-only local processes. Not an issue on the Linux g5.
- **uv offline:** `--with pyarrow/xgboost/torch` fails (not in the uv cache), but the base pyenv 3.13.9 interpreter already has them.
- **Earlier failures, now resolved:**
  - `git fetch` failed with `Permission denied (publickey)` (SSH alias `github.com-personal`); later fixed by the user.
  - AWS reads from the laptop's `default` profile (account `339712821852`) are AccessDenied on both buckets.
- **Kaggle writeup pages are JS-rendered** (WebFetch returns nothing); a zhihu summary page returned 403; local PDF rendering is unavailable (no poppler).

## 8. Current Bugs / Unresolved Issues
1. **Working tree is on stale `sai`.**
   - Expected: work on `v2/decision-layer`.
   - Actual: HEAD `sai` @ 5231dc4, behind `origin/sai` by 7; the untracked v2 files are present but can't import (`ber.decision`, `ber.tracking`, `ber.config` are missing on old `sai`).
   - Fix: `git switch v2/decision-layer` (the untracked files carry over).
2. **Nothing committed.** `v2_eval.py`, `test_v2.py`, `v2.md`, `research-v2.md`, `understanding.md` exist only on disk. Commit on `v2/decision-layer` (ask the user first; they approve commits and pushes).
3. **Harness not run on real data.** Needs the g5 (no laptop access) and, for holdout/drift, the lead's new exports.
4. **Pending the lead's deliverables:** `holdout_s1.parquet` (~150k), `pair_p_{train_all,holdout}.parquet`, test `pair_p.parquet`, all-train-S1 features (density fix), and the blocking upgrade.
5. **`aws/notebook/jobs/v1_*.sh`** (untracked) target the obsolete v1 CLI. Do not queue them; write new v2 job scripts using v0's `stages` + `v2_eval`.
6. **`aws/ssh/`** (untracked, 2 entries, created by the user) was not inspected. It may hold key material: **never commit or print it**.
7. **Local test env lacks DuckDB and MLflow.** v0's own `tests/test_v0.py`/`test_block.py` can't run locally; `PID_BASE` equality is only checked where DuckDB exists (the g5).

## 9. Git / Test / Build State
- **Branch:** `sai` (local, 5231dc4), upstream `origin/sai` at fd725b2 (behind 7).
- **Other local branches:**
  - `v2/decision-layer` (fd725b2 = `origin/sai`, no own commits, tracks `origin/sai`)
  - `v1/record-centric` (5231dc4, no own commits)
  - `er`, `main`, `0a`, `rm` (a6b842b)
- **Stashes:** `stash@{0}` "On sai: er: entity-resolution pipeline WIP" (obsolete `er` pipeline + `ml-challenge.pdf`); `stash@{1}` "On er: barani v0" (`.gitignore` tweak).
- **Untracked:**
  - `archive/`, `aws/notebook/jobs/`, `aws/ssh/`
  - `code/business_entity_resolution/src/ber/stages/` (only `v2_eval.py` + `__pycache__` on this branch)
  - `code/business_entity_resolution/tests/test_v2.py`
  - `plan.local-old.md`, `prob_statement.pdf`, `research-v2.md`, `understanding.md`, `v1.md`, `v2.md`, `HANDOFF-v2.md` (this file)
- **Checked at checkpoint time:** none of the untracked files is tracked on `origin/sai`, so `git switch v2/decision-layer` will not hit an "untracked file would be overwritten" error. The lead's lowercase `handoff.md` exists only in git on `origin/sai`, not on disk here.
- **Tests (last run, on `v2/decision-layer`):** `tests/test_v2.py tests/test_text.py`: 16 passed, 1 skipped. v0's DuckDB-based tests were not run locally.
- **No lint or type-check** configured or run for v2 files.

## 10. Important Files and Code Locations
| Path | Purpose |
|---|---|
| `v2.md` | **Plan of record v2.1**: read first |
| `research-v2.md` | model and approach research with licences and sources |
| `understanding.md` | problem / metric / data explainer (some numbers superseded) |
| `v1.md` | superseded v1 design (history only) |
| lead's `handoff.md`, `context.md`, `plan.md`, `research.md` §12, `submission_checklist.md` (on `origin/sai`) | infrastructure, data facts, v0 design, v0 error analysis |
| `code/business_entity_resolution/src/ber/stages/v2_eval.py` | evaluation harness (see §4 for symbols) |
| `code/business_entity_resolution/tests/test_v2.py` | harness tests; `_fake_work()` builds a tiny v0-layout WORK tree |
| `src/ber/decision.py` (v0) | `macro_f05`, `assign_exclusive`, `tune_threshold` |
| `src/ber/stages/{sample,pairs,train_gpu,predict,block}.py` (v0) | stages v2 extends or consumes |
| `archive/v1/` | v1 code to port: `canon.romanize` (tested Indic romaniser), `forensics._align/_subs_map`, `evaluate.bootstrap`, `test_v1.py::_brute` (expected-F0.5 brute force) |
| `plan.local-old.md` | an older local copy of the lead's `plan.md` (safe to delete once confirmed) |

## 11. Configuration
- **Env var names:**
  - `BER_WORK` (work dir), `BER_DATA` (raw TSV dir), `BER_PARAMS` (params YAML override), `BER_GIT_SHA` (tracking)
  - local test only: `PYENV_VERSION=3.13.9`
- **AWS:** region `us-east-1`; the lead's CLI profile `hxman-26` (root session; **not** to be used from this laptop). This laptop's `default` profile is a different account with no access.
- **Buckets:** data `ml-challenge-nooglers`; working `sagemaker-us-east-1-567503593043` (`ber/code`, `jobs/{pending,live,done}`, `runs/`).
- **Local test command** (offline, from `code/business_entity_resolution/`):
  `PYENV_VERSION=3.13.9 uv run --offline --no-project --python 3.13 --with rapidfuzz python -m pytest -q tests/test_v2.py tests/test_text.py -p no:warnings`
- **Local env:** base pyenv 3.13.9 has pandas 2.3.3, numpy 2.4.0, polars 1.42.1, xgboost 3.2.0, pyarrow 23.0.1, torch 2.12.1, sentence-transformers 5.4.1, pyyaml. **Missing:** duckdb, mlflow, lightgbm.
- **On the g5:** `BER_WORK=/home/ec2-user/SageMaker/work python -m ber.stages.v2_eval --run v0` (dev section works with today's v0 outputs).

## 12. Testing and Verification
- **Run:** harness unit + CLI tests (above), all green. The fake-data CLI output was checked by hand (dev 0.5, oracle 1.0, loss matcher 0.5; drift alert for an unseen country).
- **Not run:**
  - anything on real data
  - v0 tests locally (no DuckDB)
  - the DuckDB `PID_BASE` equality test
- **Remaining:** run `v2_eval --run v0` on the g5 to reproduce the lead's 0.9551 OOF as the dev number (sanity), then the holdout once it is exported.

## 13. Conversation-Only Context
- **User preferences:**
  - Wants plain-language explanations with concrete examples (said "I don't understand" several times).
  - Reviews plans before approving and often rejects plan-approval popups to review first.
  - Approves commits and pushes explicitly; earlier denied a command that combined `git fetch` + AWS calls.
  - Wants deep research (open-source, Hugging Face) before designing.
- **Team decision:** one serial track (the user + Claude). The lead owns v0 `stages/` and blocking; we own the v2 layers and the harness.
- **The lead's second review** (verbatim substance; accepted into `v2.md`): the corrected numbers, the new order, the density flaw, v0 naming, test p saved by the lead, the disjoint ~150k holdout, the France drift monitor, run names `<layer>-<MMDD-HHMM>`, scheduling the single-GPU queue.
- **Two questions from the lead are still unanswered by the user:**
  1. A scoped IAM user for this laptop vs the lead running our jobs.
  2. OK to add the holdout + `pair_p` exports now (recommended yes).
- **Suggested low-cost extra (from v1, endorsed by the lead):** an empty submission to read the test singleton share from the leaderboard.
- **The lead wants the v0 leaderboard score reported** to calibrate OOF vs LB.
- **The user was asked, with no answer yet:** "Do you want me to commit this to `v2/decision-layer` first, then start the stage-2 stacking model?"

## 14. Immediate Next Action
1. `git switch v2/decision-layer` (the untracked v2 files carry over), then run the local test command in §11: expect 16 passed, 1 skipped.
2. Ask the user to approve committing `v2_eval.py`, `test_v2.py` (+ `v2.md`, `research-v2.md`, `understanding.md` if they want them tracked) to `v2/decision-layer`. Do not push without approval.
3. Relay the lead's two open questions to the user if still unanswered.

## 15. Remaining Plan
1. **Immediately:** commit the harness; run `v2_eval --run v0` on the g5 (via the lead or a scoped IAM user) and confirm dev ≈ 0.9551.
2. **Step 2, `stages/v2_stack.py`:** stage-2 XGBoost on v0 features + p1 + consensus features:
   - S1-side share/mean/max p1 of the other records
   - similarity to the S1's best record
   - sibling digit/house/PIN agreement
   - record-side gap to the best other S1 and the count of S1 with p1 ≥ 0.5
   
   Trained out-of-fold on the dev sample, using all-S1 train p1. Writes `WORK/v2/<run>/p2_{split}.parquet` and `selected_*.parquet`; judged by `v2_eval --selected … --baseline v0`.
3. **Step 3, `stages/v2_decide.py`:** isotonic calibration, entity P(any), expected-F0.5 prefix DP (re-create from v1; brute-force test), vetoes.
4. **Step 4, `stages/v2_xenc.py`:** 30-min zero-shot pilot (gte-multilingual-reranker or mMiniLM) → mmBERT-small fine-tune (BCE, mined hard negatives + distractors, non-holdout pairs only) → band inference → feature in step 2. Schedule it against the lead's GPU jobs.
5. **Step 5:** record owner-or-none only if it beats step 3.
6. **Step 6:** sibling-expansion recall measurement (hand to the lead); transliteration dictionary; dense retrieval only if India lags.
7. **Sunday:** freeze, reproduce from scratch, methodology document, zip, official validator PASS, final submissions (≤ 5 per day, each logged).

## 16. Things the Next Session Must NOT Do
- Do not rebuild or rename v0's `prepare`/`block`/`pairs`/`train_gpu` or its ids (`q`, `pid`).
- Do not revive v1 (`v1/record-centric`, `archive/v1/`) as a pipeline, record-centric retrieval, Modal, FAISS, or a dense backbone.
- **Do not write a root `HANDOFF.md`:** case-insensitive filesystem, and it clobbers the lead's `handoff.md`.
- Do not download datasets or models on the laptop.
- Do not ask for, print or store secrets; never use or request the lead's root profile.
- Do not commit `aws/ssh/`, `prob_statement.pdf` or anything under `archive/` without the user's say-so.
- Do not queue the `aws/notebook/jobs/v1_*.sh` scripts.
- Do not commit, push, or upload to S3 without explicit user approval.
- Do not import torch in a local process that also uses xgboost/sklearn (macOS segfault).
- Do not tune on the public leaderboard; ship a layer only on a paired holdout CI > 0.

## 17. Resume Instructions
1. **Read:**
   1. `HANDOFF-v2.md`
   2. `v2.md`
   3. the lead's `handoff.md` and `research.md` §12 (`git show origin/sai:handoff.md`)
   4. then `research-v2.md` if needed
2. **Inspect:**
   - `git branch --show-current`; `git status`
   - `git log --oneline -5 origin/sai` (fetch first if the user allows)
   - `code/business_entity_resolution/src/ber/stages/v2_eval.py`, `tests/test_v2.py`
3. **Switch** to `v2/decision-layer` and run the local tests (§11).
4. **Continue** with §14, then §15 in order. For the stacking model, read `stages/pairs.py` (feature names; `blocking_features` competition columns `rank_p`, `margin_p`) and `stages/train_gpu.py` (fold handling, `pick_device`) before writing `stages/v2_stack.py`.
5. **Communicate** in plain language, check in before commits, pushes and plan approvals, and keep answers concise.
