# Handoff: v6 (`bs` line) on the barani GPU notebook

Owner: Baranidharan. Branch `v6/strong-parts` (built on `sai` @ c9a82b8). Last update: **26 Sep 2026, about 05:20 IST**.
Window closes Sun 27 Sep 23:59 IST; planned freeze **Sun 12:00 IST**.

## 1. Goal and why
- Portal (leaderboard) **above 0.98**. The team's best is `s6`: locked holdout 0.9831; the portal has read 0.012 to 0.018 below
  the holdout every time (`s4` 0.9708 on the holdout, 0.953 on the portal).
- Two levers, both needed:
  - raise the holdout to about 0.988 to 0.991 (stronger encoder, cross-encoder, set model);
  - shrink the holdout-to-portal gap to about 0.008 (train the first stage and the stack at the test's pool density).
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
stack               XGBoost + xs, **trained at test density (--thin)**                             (bs_s6 -> **bs_w2**)
**set model**       3-layer transformer over each S1's shortlist, stacked on the stack's OOF p -> blended with the stack
decision            one record -> one S1, threshold tuned out of fold, **cap 5 S2 + 6 S3 per S1**   (**bs_final**)
```
Evaluation:
- Locked holdout: 150k S1, `split.holdout_q`. Every step must beat the previous one on a paired bootstrap interval.
- **Final holdout** (`split.final_q`, 100k S1): never trained on and never looked at. Checked once, at the freeze, with
  `src/scripts/final_check.py`, to catch overfitting to the locked holdout after many versions.
- Test-density thinning (`split.thin_q`, params `thin.frac`): drops a fraction of train S1 (never holdout, final or sample
  S1), so their true records become ownerless, as the test's extra pool records are. `src/scripts/density_check.py` measures
  whether the extra test records really are such orphans, and how many.

## 3. Code map (all in `code/business_entity_resolution/`)
| Piece | Files |
|---|---|
| State names | `src/ber/text.py`, `stages/block.py` (INDEX_VERSION 5) |
| Cap | `src/ber/decision.py` (`cap_per_source`), `stages/stack.py` (kept only if not worse on the holdout), `stages/predict.py` |
| Final holdout, thinning | `src/ber/split.py` (`final_q`, `unscored_q`, `thin_q`), `stages/pairs.py --thin`, `stages/stack.py --thin` |
| Second encoder | `stages/dense_all.py --tag 2` (params `dense_all2`), `src/ber/decoys.py` |
| Cross-encoder | `stages/xenc.py` (params `xenc`), `stages/stack.py --xenc` |
| Set model and blend | `stages/setmodel.py` (`train`, `blend`) |
| Checks | `src/scripts/density_check.py`, `src/scripts/final_check.py` |
| Pipeline targets | `Makefile`: `bs_prepare bs_block bs_dense bs_dense_all bs_first bs_stack` (baseline), then `bs_density bs_thin_stack bs_encoder bs_xenc bs_final_stack bs_set` (`THIN=--thin` from the environment) |
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
| Job | What | Status / result |
|---|---|---|
| `a1_base` | baseline rebuild, stages 1 to 2 | prepare 6 min, block 25 min (test 170.7M raw pairs, train 216.6M); pruner top-30 pair recall **0.9479** (team 0.9471); then failed at `bs_dense`: `WORK/dense/` missing on a fresh box (fixed, commit 7d11e81) |
| `a2_density` | density check | failed harmlessly (ran before the baseline existed); re-queued as `a2b_density` |
| `a1b_resume` | baseline stages 3 to 6 | **running**. Name encoder fine-tuned in 104 s (91,969 S1, same as the team). Name+address encoder: 724 s fine-tune; holdout pairs missed by token search 23,865 (team 24,230), recovered 15,649 (team 16,025); added 1.44M test / 1.07M train pairs. First stage `bs_v5` **OOF F0.5 0.9763** (team `v5` 0.9757). Stack `bs_s6` expected about 06:30 IST |
| `a2b_density` | density check + stack at test density (`bs_s6t`) | queued |
| `b1_encoder`, `c1_xenc`, `d1_stack` | second encoder + `bs_w1`; cross-encoder; stack `bs_w2` + set model + blend `bs_final` | job files ready, queued after the density result (`THIN` set from it) |

Expected (estimates, to be replaced by measurements):

| Model | Locked holdout | Portal |
|---|---|---|
| `bs_s6` | about 0.983 | about 0.966 to 0.971 |
| `bs_final` | about 0.988 to 0.991 | about 0.973 to 0.982 (central 0.978) |

## 6. Next steps
1. Read `bs_s6`'s holdout: it must reproduce about 0.983; otherwise fix the rebuild first.
2. Read `density_check`: the orphan share and the thinning fraction. Set `THIN` for `b1` and `d1`, and `thin.frac` in params
   if the measured fraction differs from 0.19.
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
