# Handoff: v6 (`bs` line) on the barani GPU notebook

Owner: Baranidharan. Branch `v6/strong-parts` (built on `sai` @ c9a82b8). Last update: **26 Sep 2026, about 06:45 IST**.
Window closes Sun 27 Sep 23:59 IST; planned freeze **Sun 12:00 IST**.

## 1. Goal and why
- Portal (leaderboard) **above 0.98**. The team's best is `s6`: locked holdout 0.9831; the portal has read 0.012 to 0.018 below
  the holdout every time (`s4` 0.9708 on the holdout, 0.953 on the portal).
- Two levers, both needed:
  - raise the holdout to about 0.988 to 0.991 (stronger encoder, cross-encoder, set model);
  - shrink the holdout-to-portal gap to about 0.008. Density check (26 Sep): the test's extra ownerless records are look-alike
    **decoys**, not orphans, so the lever is telling decoys apart (cross-encoder, set model, decoy features) plus a threshold
    tuned for the test's decoy density (`test_weighted.py`). Thinning was tried and removed.
- The best possible holdout on today's candidates is 0.9953, so the matcher alone cannot reach 0.98 on the portal.

## 2. Architecture (changes against `s6` in bold)
```
prepare             text cleaning + **state names -> codes (US, India, Indic scripts)**
candidates          token cascade (K100 -> pruner 30) U name encoder U name+address encoder (dense_all)
                    + **dense_all2: second e5-small channel, ~900k unused S1, 3 hard negatives + 1 synthetic decoy per S1**
first XGBoost       67 features + **dall2_cos / dall2_rank**                                       (bs_v5 -> **bs_w1**)
shortlist           best 10 per S1 by p1 (p1 >= 0.005)
**cross-encoder**   gte-multilingual-reranker-base on 0.05 <= p1 <= 0.95; input "S1 [SIB] its most confident other
                    record" vs candidate; synthetic look-alikes as negatives -> feature xs
stack               XGBoost + xs + **decoy edit features + dall2 columns**                          (bs_s6 -> **bs_w2**)
**set model**       3-layer transformer over each S1's shortlist, stacked on the stack's OOF p -> blended with the stack
decision            one record -> one S1, threshold tuned out of fold, **cap 5 S2 + 6 S3 per S1**   (**bs_final**)
```
Evaluation:
- Locked holdout: 150k S1, `split.holdout_q`. Every step must beat the previous one on a paired bootstrap interval.
- **Final holdout** (`split.final_q`, 100k S1): never trained on and never looked at. Checked once, at the freeze, with
  `src/scripts/final_check.py`, to catch overfitting to the locked holdout after many versions.
- `src/scripts/density_check.py`: are the test's extra ownerless records orphans or look-alike decoys? (Answer: decoys.)
- `src/scripts/test_weighted.py MODEL [--apply]`: holdout score post-stratified to the test's mix of country x look-alike
  density (label-free cells), and the threshold re-tuned for that mix on out-of-fold S1 only; `--apply` writes `MODELtw`.

## 3. Code map (all in `code/business_entity_resolution/`)
| Piece | Files |
|---|---|
| State names | `src/ber/text.py`, `stages/block.py` (INDEX_VERSION 5) |
| Cap | `src/ber/decision.py` (`cap_per_source`), `stages/stack.py` (kept only if not worse on the holdout), `stages/predict.py` |
| Final holdout | `src/ber/split.py` (`final_q`, `unscored_q`); stack and set model never train on it |
| Second encoder | `stages/dense_all.py --tag 2` (params `dense_all2`), `src/ber/decoys.py` |
| Cross-encoder | `stages/xenc.py` (params `xenc`), `stages/stack.py --xenc` |
| Set model and blend | `stages/setmodel.py` (`train`, `blend`) |
| Checks | `src/scripts/density_check.py`, `src/scripts/test_weighted.py`, `src/scripts/final_check.py` |
| Pipeline targets | `Makefile`: `bs_prepare bs_block bs_dense bs_dense_all bs_first bs_stack` (baseline), then `bs_density bs_tw bs_encoder bs_xenc bs_final_stack bs_set` |
| Tests | `src/tests/test_v6.py`, `test_decoys.py` (run with `OMP_NUM_THREADS=1` on a Mac: torch and XGBoost clash there) |

## 4. AWS (account 645311222213, profile `barani`, us-east-1)
- **Notebook:** `barani-v5`, ml.g5.16xlarge (64 vCPU, 256 GB, one A10G), 200 GB volume, lifecycle config `barani-queue`.
  Stops itself after 1 h with no job running (job-aware).
- **Queue:** bucket `sagemaker-us-east-1-645311222213`.
  - Code is published to `ber/code`.
  - Jobs go to `jobs/pending/` and run one at a time, alphabetically.
  - Logs appear in `jobs/live/` while running and in `jobs/done/` after (last line `exit=`).
- **Checkpoints:** models, reports and outputs of each stage go to
  `s3://ml-challenge-nooglers/ml-challenge-2026/checkpoints/barani/bs/NN-<stage>-<time>/`, with a `manifest.json`.
- **Laptop commands:** `smssh-venv/bin/python aws/sm/sm.py --profile barani <cmd>`:
  - `nb start|stop|status`;
  - `publish` (syncs the package and `entry.py`);
  - `enqueue aws/queue/jobs/<job>.sh --name <name>`;
  - `jobs`;
  - `jlog <name> [--no-follow]`.
- **Credentials:** the AWS login expires periodically. Run `aws login --profile barani` in your own terminal. Never paste keys,
  presigned URLs or Jupyter links in chat or git.

## 5. Status and results
| Job | What | Result |
|---|---|---|
| `a1_base` + `a1b_resume` | baseline rebuild of `s6` | **`bs_s6` locked holdout 0.98299** (team `s6` 0.9831): reproduced. First stage `bs_v5` holdout 0.9763 [0.9758, 0.9767] (team 0.9757); pruner top-30 recall 0.9479 (team 0.9471); dense_all recovers 15,649 of 23,865 missed pairs. Stack gain over first stage +0.0067 [0.0064, 0.0071]. Cap 5+6: no change (0.98299 both). Official validator PASS. 134 min. One fix on the way (`WORK/dense` missing on a fresh box). |
| `a2b_density` | orphans or decoys? | pool per S1 train 4.68, test 5.75; ownerless per S1 train 1.22, test unclaimed 2.32. Nearest-S1 cosine: train decoys 0.670, simulated orphans 0.562, **test unclaimed 0.671 -> orphan share 0: the extra test records are decoys**. Thinned stack `bs_s6t` 0.98294 (no change). Thinning removed. |
| `b1_encoder` | dense_all2 (e5-small, 900k S1, 3 hard negatives + 1 synthetic decoy) + first stage `bs_w1` | running (7,030 fine-tuning steps, then embed / retrieve / features) |
| `b2_tw` | test-weighted holdout and threshold for `bs_s6` | queued |
| `c1_xenc`, `d1_stack` | cross-encoder; stack `bs_w2` (+ xs, decoy features, dall2 columns) + set model + blend `bs_final` | queued |

Local checks: `pytest` 34 pass; a CPU smoke run of the whole v6 chain on synthetic data (scratch script, not in the repo) found
and fixed a set-model bug (final-holdout rows were dropped).

Expected (estimates, to be replaced by measurements):

| Model | Locked holdout | Portal |
|---|---|---|
| `bs_s6` | 0.98299 (measured) | about 0.966 to 0.971 |
| `bs_final` | about 0.988 to 0.991 | about 0.973 to 0.982 (central 0.978) |

## 6. Next steps
1. Done: `bs_s6` reproduces `s6` (0.98299).
2. Read `b2_tw`: the test-weighted estimate of `bs_s6` and whether a re-tuned threshold helps under the test's decoy density.
3. Queue `b1_encoder`, then `c1_xenc`, then `d1_stack`. After each one: paired result on the holdout, and an error analysis
   (`src/scripts/error_analysis.py`) to pick the next change.
4. Backlog after `d1`, by expected gain:
   - e5-base for dense_all2;
   - more first-stage training S1;
   - a larger cross-encoder (bge-reranker-v2-m3);
   - seed or LightGBM ensemble of the stack;
   - a second consensus round.
5. Freeze Sun 12:00 IST:
   - `final_check.py` on the pick;
   - checker and official validator;
   - hand `output/bs_final/` and the feature parquets (`xs`, `dall2_*`) to the team.

Needs from the team (Hariheman holds the portal login):
- `s6`'s portal score;
- one upload of `s6` with all France predictions emptied (the difference gives France's real F0.5);
- agreement on using `bs_final` in the team submission.

## 7. Known issues
- On the Mac, torch plus XGBoost in one process segfaults unless `OMP_NUM_THREADS=1`. On the notebook this is not a problem.
- `src/tests/test_v5.py` and the untracked `keys.py`, `expand.py`, `channels.py`, `measure.py` are left over from the
  earlier v5 branch. They are not part of v6 and not committed.
- Model names use the `bs_` prefix so nothing collides with the team's `v*` / `s*` / `z*` runs.
