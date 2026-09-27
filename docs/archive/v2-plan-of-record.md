> **ARCHIVED (26 Sep 2026): Early plan of record (25 Sep); the built pipeline differs, see ARCHITECTURE.md and EXPERIMENTS.md. Kept for history only; do not follow its instructions.**

# v2.1: plan of record

Written 2026-09-25. It supersedes `v1.md` and the earlier `v2.md` drafts. It is the v2 design with the v0 owner's review applied:
1. corrected numbers
2. order by measured loss
3. the density fix
4. v0 naming in the contracts

Background: `understanding.md`, `research-v2.md`, `research.md` §12 (v0 error analysis), `handoff.md`.

## Final architecture at a glance

```
STAGE 1  CANDIDATES      v0 token-index blocking, top-K per S1          (lead; upgrade: new token types, larger K, pruner)
STAGE 2  PAIR SCORING    2a  v0 features + XGBoost → p1                  (lead; scored for ALL train S1, not just the sample)
                         2b  cross-encoder (mmBERT-small) on uncertain pairs only → xs   (us, gated ≥ +0.005)
                         2c  stage-2 XGBoost: p1 + xs + sibling-agreement + record-side competition features → p2  (us)
STAGE 3  DECISION        isotonic calibration of p2
                         [record owner-or-none model: only if it beats the step below]
                         per-S1: P(any match) + keep the set with the best expected F0.5 + postcode/house-number vetoes  (us)
STAGE 4  OUTPUT          matching_results.tsv, candidate_pairs.tsv, official validator
ACROSS   EVALUATION      locked 150k holdout + bootstrap CI, breakdowns, loss split, US↔India transfer, train-vs-test check,
                         France drift monitor on test, LB log
```

**Where the record-centric ideas live:**
- in 2c: record-side competition features, and full competitor lists thanks to the density fix
- optionally in stage 3: the owner-or-none model

A layer ships only if it beats the previous best on the locked holdout by more than the bootstrap CI (~0.003).

## 1. Current numbers (v0 after the feature pass)

| Metric | Value |
|---|---|
| OOF macro F0.5 | **0.9551** (India 0.935, non-Latin 0.897 vs oracle 0.920, singletons 0.959) |
| Total loss vs 1.0 | 4.5 points = **blocking recall 2.19** + **matcher 2.30** |
| Exclusive assignment | Same score as plain (0.9551): the competition features `margin_p` and `rank_p` (the top two features) already encode "another S1 explains this record better" |
| Threshold curve | Flat: 0.9532 at 0.50, 0.9550 at 0.63 |
| False positives | 84% look-alike distractors (near-copies with 1–2 changed characters or digits) |
| Non-Latin remaining gap | 2.3 points on ~11% of entities → at most ~0.25 overall |

**Realistic headroom on train-like data: about 1.5 to 2.5 points in total:**
- blocking ~1–1.5
- stacking ~0.5–1
- decision ~0.3

## 2. Priorities by measured loss

| # | Layer | Owner | Targets | Expected | Gate |
|---|---|---|---|---|---|
| 1 | **Blocking upgrade** (token types g, x, k; larger K; first-stage pruner) | lead | blocking loss 2.19 | ~1–1.5 | recall ↑ at fixed cost |
| 2 | **Stage-2 stacking with consensus features** (2c) | us | look-alike distractors (84% of FPs) | ~0.5–1 | holdout > CI |
| 3 | **Decision + calibration** (isotonic; entity P(any); per-S1 expected F0.5; vetoes) | us | singleton vs matched trade-off, lone weak candidates | ~0.3 | holdout > CI |
| 4 | **Cross-encoder** on the uncertain band (2b; mmBERT-small, MIT) | us | residual digit/typo look-alikes, non-Latin | gated | ≥ +0.005 and ≤ 60 min inference |
| 5 | **Record owner-or-none** (none option, calibrated r) | us | overlaps with `margin_p`/`rank_p`: likely < 0.3 | only if it helps | holdout > CI |
| 6 | **Transliteration dictionary**, then **dense retrieval** only if India lags | us | non-Latin (≤ ~0.25) | last | — |
| — | **Sibling-expansion measurement** | us → lead | blocking misses recoverable via siblings | feeds #1 | build only if ≥ 0.5 point |

## 3. Stage-2 consensus stacking (the most valuable matcher layer)

Stage-1 p1 comes from v0. Stage 2 is a GBM on v0's features + p1 + the features below, all using out-of-fold p1 on train and computed identically at test:
- **S1 side:** number and share of the S1's candidates with p1 ≥ 0.5; mean and max p1 of the S1's *other* records; similarity of this record to the S1's best record (name, digit string, house number).
- **Sibling agreement on digits and names:** does this record's house number, PIN and digit string agree with the S1's confident records? A near-copy distractor (`312` vs `323`) disagrees with the real siblings.
- **Record side:** p1 gap to the record's best *other* S1; how many S1 list this record with p1 ≥ 0.5.
- (Later) the cross-encoder score `xs` on the band.

## 4. The density fix

- **Problem.** Train p1 exists only for the 250k sampled S1 (~11% of 2.2M). Record-side competition features computed from that subset understate competition. The model would train in a sparse world and deploy in a crowded one (worst for France).
- **Fix:**
  - **Score all train S1 pairs** (~66M), so every record's full competitor list has p1, as at test.
  - S1s outside the sample were never trained on, so the sample-trained model gives unbiased p1 for them.
  - The 250k sample keeps its OOF p1.
  - This needs the lead's vectorised features.

## 5. Data contracts (v0 naming; do not rename)

v0 ids:
- `q` = S1 rid.
- `pid` = src × 10,000,000 + record rid (S2 = 2, S3 = 3).
- `WORK` = `BER_WORK`.

| File | Writer | Columns | Status |
|---|---|---|---|
| `WORK/parquet/{split}/source{1,2,3}.parquet`, `labels.parquet` | lead `prepare` | canonical columns; `s1_rid, src, other_rid` | exists |
| `WORK/blocks/{split}/cand_*.parquet` | lead `block` | `q, pid, score, s_<type>, ns` | exists; upgrade #1 changes content, not schema |
| `WORK/sample/train_s1.parquet` | lead `sample` | `rid, fold, n_matches, ctry` (250k dev, 5 folds) | exists (dev may grow) |
| `WORK/sample/holdout_s1.parquet` | lead `sample` | `rid, n_matches, ctry` (~150k S1, **disjoint**, from the remaining 1.95M) | **new**: locked holdout, features only; never trained on |
| `WORK/features/train/part_*.parquet` | lead `pairs` | `q, pid, [label], features` | **extend** to all train S1 (density fix) |
| `WORK/models/<run>/oof.parquet` | lead `train_gpu` | `q, pid, p, label` for the dev sample (OOF) | exists |
| `WORK/models/<run>/pair_p_{train_all,holdout}.parquet` | lead | `q, pid, p` for non-sample S1 and holdout (final model) | **new** |
| `WORK/output/<run>/pair_p.parquet` | lead `predict` | `q, pid, p` (test) | **new** |
| `WORK/v2/<run>/{p2,xenc,assign,entity}_{split}.parquet` | us | `q, pid, p2` / `q, pid, xs` / `pid, q, r` / `q, p_any, n_selected` | new |
| `runs.jsonl` + MLflow (`tracking.log_stage`) | all | layer, holdout F0.5 + CI, breakdowns, LB score | reuse |

Run names: `<layer>-<MMDD-HHMM>` (e.g. `stack-0926-1030`).

**Leakage rules:**
- Nothing is trained on holdout S1s.
- Every train p that feeds a later layer is out-of-fold (sample) or from a model that never saw that S1 (non-sample, holdout).
- The dictionary and the cross-encoder learn only from non-holdout pairs.
- Consensus features use those p only.

## 6. Evaluation harness (us)
- **Scores:** holdout macro F0.5 (vectorised `decision.macro_f05`) with a bootstrap 95% CI.
- **Breakdowns:** country, script, singleton vs 1 / 2–3 / 4+ matches, empty address, shared address.
- **Loss decomposition:** blocking vs matcher, via an oracle on the candidates.
- **US↔India transfer:** train on one country, score the other.
- **Adversarial train-vs-test check:** a classifier that tries to tell train pairs from test pairs.
- **France drift monitor on test:** per-country predicted match rate, mean p, and the share of S1s with ≥ 1 match vs the train 94.4%.
- **Leaderboard log:** LB score per submission next to holdout; an optional empty submission gives the test singleton share.

## 7. Operations
- **One GPU, one sequential queue.** The cross-encoder fine-tune (~1 h) is scheduled against the blocking and feature runs.
- **Access.** No root login is shared. Either a scoped IAM user (working bucket `jobs/`, `runs/`, `ber/` prefixes + read on the data bucket; keys created by the user outside chat), or the lead runs our jobs.
- **Branch** `v2/decision-layer` from `origin/sai`: new modules only; the lead keeps `stages/`.

## 8. Our build order (serial)

| Step | Work | Depends on |
|---|---|---|
| 1 | Evaluation harness over the dev OOF and the holdout (CI, breakdowns, loss split, drift monitor) | holdout + pair_p exports (lead) |
| 2 | Stage-2 consensus stacking → holdout > CI → submission | all-S1 train p1 (lead) |
| 3 | Decision: isotonic calibration, entity P(any), expected-F0.5 prefix, vetoes | 2 |
| 4 | Cross-encoder: 30-min zero-shot pilot → mmBERT-small fine-tune → band inference → feature in stage 2 | 2; GPU slot |
| 5 | Record owner-or-none, only if it beats step 3 alone | 3 |
| 6 | Sibling measurement (to the lead); dictionary; dense retrieval only if India lags | anytime |
| Sun | Freeze, reproduce, methodology document, zip, official validator PASS, final submissions | — |

## 9. Files (new modules on `v2/decision-layer`; knobs in `configs/params.yaml`)

| Module | Content | Reuse |
|---|---|---|
| `stages/v2_eval.py` | harness + drift monitor | `decision.macro_f05`, `tracking.log_stage`, `scripts/error_analysis.py` logic |
| `stages/v2_stack.py` | consensus features + stage-2 XGBoost, OOF on dev | `train_gpu.pick_device`, v0 feature parts |
| `stages/v2_decide.py` | calibration, entity GBM, expected-F0.5 prefix, vetoes (+ owner-or-none when enabled) | `decision.assign_exclusive` as the baseline |
| `stages/v2_xenc.py` | cross-encoder fine-tune + band inference | — |
| `stages/v2_siblings.py`, `stages/v2_translit.py` | sibling measurement; dictionary | v1 romaniser |
| `tests/test_v2.py` | expected F0.5 vs brute force, consensus features on synthetic data, no-leak assertions | — |

## 10. Verification
- `pytest` green (v0 tests + `test_v2.py`) before any sync to S3.
- Every layer: one queued job writes its Parquet + `report.json`. The harness prints holdout macro F0.5 with 95% CI, breakdowns and the loss split against the previous best run.
- Before upload: official validator PASS; matched ⊆ candidates; drift monitor within bounds; LB logged next to holdout.

## 11. Open items (the lead's questions)
1. Access: scoped IAM user (recommended) or the lead runs our jobs.
2. The disjoint ~150k holdout and the `pair_p` exports: recommended now (they unblock steps 1 and 2).
