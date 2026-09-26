# Architecture reference: current pipeline, every earlier version, and what was tried

Team Nooglers, Amazon ML Challenge 2026 (Business Entity Resolution). Written 26 Sep 2026 about 02:15 IST, updated through 13:00 IST (the current best is `s17`; experiment registry in `EXPERIMENTS.md`). This is the single place that describes what the system is, how it got there, and why each piece exists. Companion documents: `TEAM_GUIDE.md` (how to run things, AWS how-to, backlog), `handoff.md` (infra details and latest status, section 0), `context.md` (data facts), `research.md` (measurements, sections 10 to 19), `plan.md` (the original v1 plan), `submission_checklist.md`, `submission/Documentation_template.md` (methodology draft for the organisers).

## 1. Problem and data

**Task.** For every Source 1 (S1) business record, predict which Source 2 (S2) and Source 3 (S3) records describe the same real-world business. Score: macro F0.5 over S1 entities (precision counts twice as much as recall; an S1 with no match scores 1.0 only if the prediction is empty, and 0 if it predicts anything). Two files are submitted: `matching_results.tsv` (the predictions) and `candidate_pairs.tsv` (the pairs the model scored; every matched id must be inside it). The organisers also rank on the candidate file: a smaller candidate set per S1 ranks higher.

**Data sizes.**

| | S1 | S2 | S3 |
|---|---|---|---|
| train | 2,206,821 | 5.03M | 5.29M |
| test | 1,732,544 | 4.89M | 5.08M |

Test S1 by country: US 663,106 (38.3%), India 809,986 (46.8%), France 259,452 (15.0%). Train has only US (60%) and India (40%): France has no labels.

**Structure that the design exploits (measured, `context.md` section 2b and `research.md` sections 10 to 14).**
- Every S2/S3 record belongs to at most one S1 (0 of 7.6M matched ids claimed twice); about 27% of train pool records (about 40% of test pool records) belong to none (distractors).
- An S1 has 0 to 11 matches (at most 5 in S2, at most 6 in S3), mean 3.46; 5.6% have none.
- Noise between the copies of one business: HTML entities, digit-for-letter swaps, typos, transposed or dropped words, injected suffix words, legal-form variants (`Pvt Ltd`, `Private Limited`), "doing business as" text and domain-style names, completely different aliases, empty or partial addresses, dropped or added house-number digits, glued words, and non-Latin names and addresses (Indian scripts, India only, about 12% of S1 have a non-Latin match).
- 84% of false positives of the first model were look-alike distractors (near copies, often with a house number that lost or gained a trailing digit).
- Entity ids are random (correlation between S1 and matched pool ids is 0.0001); test and train share no name plus address pair. There is no leak to exploit and none is used.

## 2. Evaluation protocol (used for every number in this document)

- **Stage-1 sample:** 250,000 train S1 (`sample.py`), 5 folds grouped by S1 (a fold never splits an S1).
- **Locked holdout:** 150,000 train S1 drawn once with seed 2026 from the S1 outside the sample (`ber.split.holdout_q`). No model trains on them, thresholds are never tuned on them; they are used to report and to pick between models.
- **Stage-2 training set:** 400,000 other S1 (plus the sample S1 with out-of-fold probabilities); all first-stage probabilities used there come from a model that never saw those S1.
- **Leakage guard:** S1 used to fine-tune the dense encoders are excluded from stage-2 training (their dense similarity would look better than on new data).
- **Significance:** paired bootstrap (1000 resamples of S1) on the same holdout; a change ships only if its 95% interval against the current best excludes 0 (`decision.paired_bootstrap_delta`, `src/scripts/paired_models.py`).
- **Portal:** public subset of the test set; it has read 0.012 to 0.018 below the holdout each time (test is harder: 5.8 pool records per S1 against 4.7, more India, France unlabeled).

## 3.0 Update 26 Sep 06:40 IST: the current best pipeline is `s15` (supersedes the `s5`/`s6` numbers below)

Three changes since `s6`, in order of importance:

1. **Cross-encoder on the uncertain band (`stages/xenc.py`), feature `xs`.** `multilingual-e5-small` (MIT, 118M) with a one-logit head reads both records together ("name | address", romanised name appended for non-Latin names, digit runs tagged `[N]12[/]` and `[P]60001[/]`). Fitted on 400k S1 (458k pairs: the band of p1 0.02 to 0.98, every confident false positive as a hard negative, 10% of the easy positives), 3 epochs, 32 minutes on an A10G. Only band pairs are scored (0.75 per S1 on train, 1.11 per S1 on test for `v7`), and the score enters the stack as one feature. The S1 it was fitted on are excluded from stage-two training. Effect: +0.0053 F0.5 on the holdout (`s8` 0.98362 to `s11` 0.98887), the largest single gain since the dense channels.
2. **First stage `v7`:** trained on 850k S1 (the 250k sample plus 600k S1 from the rest, never the holdout and never the S1 the dense encoders were fitted on), 5-fold out-of-fold over all of them, depth 9, eta 0.05, up to 3000 rounds (the old settings hit the 600-round cap). Holdout 0.97796 against 0.97565 for `v5`; precision 0.9933, recall 0.9527.
3. **Stack settings:** decoy edit features (`dc_*`: Levenshtein, Indel, substitution estimate, Hamming for equal lengths, length difference, first letter, word symmetric difference), more carried first-stage columns (dense channel columns, coverage, skeleton, length), 1.5M S1 for stage two, depth 9, eta 0.05, up to 1500 rounds.

```
candidates (token blocking + dense names + dense_all)  ->  first stage v7 (850k S1, depth 9)  ->  p1
p1  ->  shortlist (best 10 by p1, p1 >= 0.005)  ->  band pairs (0.02 <= p1 <= 0.98)  ->  cross-encoder xs
shortlist + xs  ->  stack s15 (consensus, digit, TF-IDF, decoy edit, carried columns, xs; 1.5M S1; depth 9)
->  exclusive assignment, threshold 0.71  ->  matching_results.tsv, candidate_pairs.tsv (4.74 candidates per S1 on test)
```

Holdout F0.5 of `s15`: 0.98935 (paired +0.00018, CI [+0.00004, +0.00032], over `s13`; `s13` 0.98917 uses base `v5`). Stack ablation trail on the way: see the version table (section 4).

## 3.1 Current architecture (`s5` / `s6`, the best models)

```
raw TSV (train, test)
  |
  v
[1] prepare        text normalisation per record: unescape HTML, strip accents (Latin only), fix digit-for-letter swaps,
  |                split "doing business as" and domain names, legal-form table, romanise non-Latin text (anyascii),
  |                country-specific address abbreviations (France rules: rue, bd, cedex ...). Output: parquet per source.
  v
[2] candidates     union of three channels, merged into WORK/blocks/{split}/cand_*.parquet
  |   (a) token blocking   DuckDB inverted index over the 10.3M pool records, IDF-weighted, token types name, address, 5-char
  |                        prefix, composite rare name+address keys, house-number+address key; df cap 800; K=100 raw per S1;
  |                        learned pruner (XGBoost on blocking features) keeps the best 30                      (v2)
  |   (b) dense, names     multilingual-e5-small fine-tuned on true India pairs; non-Latin pool names -> top-5 S1     (v3)
  |   (c) dense_all        the same encoder fine-tuned on 250k S1 of all countries with one mined hard negative per
  |                        pair; embeds "name | address" of EVERY S1 and pool record; each pool record -> its nearest
  |                        S1 (top 1, no cosine cut-off)                                                       (v5)
  v
[3] first stage    67 features per pair -> XGBoost on GPU (5-fold grouped OOF, 250k S1) -> p1        (models v5)
  |                 string similarity (name and address: ratio, partial, token sets, Jaro-Winkler, Levenshtein),
  |                 name rarity, exact-name, coverage, glued-name, house number and digit alignment, pin/postcode,
  |                 legal-form conflict, romanised and consonant-skeleton similarity, blocking scores, competition
  |                 (margin_p, rank_p: how much better another S1 explains the same record), dense cosines and ranks
  v
[4] shortlist      per S1 the best 10 pairs by p1 with p1 >= 0.005 (the best pair always kept)           (s6)
  |                 4.9 candidates per S1 on test (31 without it), no loss of F0.5
  v
[5] stack          second XGBoost on consensus features of p1 (400k S1)                                 (models s5, s6)
  |                 S1-level: confident candidates per source, rank, gap to the best; record-level: claims by other S1,
  |                 margin to the best competitor; house-number relation and digit consensus with the S1's other confident
  |                 records; TF-IDF cosine (char 3-grams of names, word 1-2-grams of addresses); similarity to the S1's
  |                 best other record; 25 carried first-stage features
  v
[6] decision       exclusive assignment (a pool record goes to its highest-probability S1), threshold about 0.67 tuned
  |                 out-of-fold; per-S1 lists of matched ids
  v
[7] outputs        matching_results.tsv and candidate_pairs.tsv (exactly the shortlist the stack scored), then the
                   official validator and src/scripts/check_submission.py
```

**Numbers for the current models (locked holdout, 150k S1).** `s5`: F0.5 0.9832 (India 0.9812, US 0.9845), precision 0.9949, recall against all true pairs 0.9652, candidates contain 98.4% of true pairs, oracle on candidates 0.9953 (blocking loss 0.0047, matcher loss 0.0121). `s6`: 0.9831, difference against `s5` -0.0001 with 95% interval [-0.0003, +0.0001], 4.9 candidates per S1 on test, candidate file 126 MB against 726 MB.

## 4. Version history (what each version added, and what it scored)

Holdout = locked 150k-S1 set; out-of-fold (OOF) = the 250k training sample; portal = public leaderboard. "Never uploaded" means no portal score exists.

| Version | Change | Score | Notes |
|---|---|---|---|
| `v0` | 42 features, XGBoost (GPU), token blocking at 30 candidates per S1 | OOF 0.9377 | never uploaded; first end-to-end run and the first output that passed the official validator (an early bug wrote empty lists as `""`; fixed with `quote_style="never"`) |
| `v1` | 63 features: name rarity, exact-name, token coverage, glued-name similarity, digit alignment, romanised names (anyascii), consonant skeletons | OOF 0.9551 (US 0.968, India 0.935) | never uploaded |
| `v2` | cascade blocking: 100 raw candidates per S1 pruned to 30 by a learned first-stage ranker (pair recall 0.9471 against 0.9415) | OOF 0.9568, holdout 0.9565 [0.9559, 0.9572], **portal 0.944** | first portal score; blocking loss 0.0195 and matcher loss 0.0188 on holdout |
| `s1` | consensus stacking: stage two on S1-level and record-level competition features of p1 | holdout 0.9617 | +0.0051 [0.0048, 0.0055] over `v2` |
| `s2` | + house-number relation and digit-consensus features (dropped or added trailing digit) | holdout 0.9671 | +0.0106 [0.0102, 0.0110]; never uploaded |
| `e1` | calibration (isotonic) plus exact per-S1 expected-F0.5 selection | +0.0001, interval includes 0 | null result; probabilities were already calibrated (ECE 0.00013); not shipped |
| `s3all` | + TF-IDF cosine and name/address consensus features (`cons_*`, `bo_p`) | holdout 0.9677, **portal 0.949** | +0.0111 over `v2`; ablations: no consensus 0.9676, no TF-IDF 0.9672 |
| `v3` | + name-only dense channel (e5-small fine-tuned on 92k India S1, top-5 non-Latin candidates), France address rules, index version 4 | OOF 0.9602, holdout 0.9598 [0.9592, 0.9604] | first-stage +0.0033 over `v2`; recall on non-Latin pairs 75.9% to 85.3% |
| `s4` | stack on `v3` | holdout 0.9708, **portal 0.953** | +0.0031 [0.0028, 0.0034] over `s3all` |
| `v5` | + `dense_all` (name and address of every record, hard negatives, pool to S1 top-1) | OOF 0.9757, holdout 0.9757 [0.9752, 0.9761] | first-stage +0.0159 over `v3`; recall 0.950, precision 0.992; top feature `dall_cos` |
| `s5` | stack on `v5` | holdout 0.9832 | +0.0075 over `v5`, +0.0124 over `s4`; portal pending |
| `s6` | `s5` with the candidate shortlist (K 10, floor 0.005) | holdout 0.9831 | 4.9 candidates per S1, the file for the final ranking; portal pending |
| `v6` / `s7` | extra dense neighbours (top 3) for pool records with an empty address | 0.97557 / 0.98309 | no gain (`v5` 0.97565, `s6` 0.98308); dropped |

| `s8` | stack with decoy edit features and extra carried first-stage columns | holdout 0.98362 | +0.00054 [0.00036, 0.00073] over `s6`; decoy features alone +0.00024 |
| `s9` | second consensus round on `s6`'s probabilities | 0.98317 | +0.00009, CI includes 0; dropped |
| `s10` | stage-two training set 1.5M S1 instead of 400k | 0.98343 | +0.00036 [0.00019, 0.00052] over `s6` |
| `s12` | decoy + extra features + 1.5M S1 + depth 9, eta 0.05, 1500 rounds | holdout 0.98505 | +0.00198 [0.00177, 0.00219] over `s6`; depth 9 beat 6 and 11 on out-of-fold and holdout |
| `s11` | `s8` + cross-encoder score `xs` | holdout **0.98887** | +0.00525 [0.00497, 0.00555] over `s8` |
| `s13` | `s11` + 1.5M S1 + depth 9 | holdout 0.98917 | +0.00030 [0.00019, 0.00041] over `s11` |
| `v7` | first stage on 850k S1 (250k sample + 600k), depth 9, eta 0.05, 3000 rounds | holdout 0.97796 [0.9775, 0.9784] | +0.0023 over `v5` |
| `s15` | stack on `v7` with cross-encoder, decoy, extra, 1.5M S1, depth 9 | holdout **0.98935** | +0.00018 [0.00004, 0.00032] over `s13`; 4.74 candidates per S1 on test; current best |

| `s14` | stack on base `v5` with cross-encoder 2 (`multilingual-e5-base`, 278M, fitted on 700k S1 = 966k pairs, band 0.01 to 0.99, 3 epochs, 2.4 h on an A10G) | holdout **0.98986** | +0.00069 [0.00054, 0.00083] over `s13`; average precision inside the band 0.986 against 0.934 for p1 (train pairs, optimistic) |
| `s16` | `s15` + competition features on the cross-encoder refined probability (`--xcons`) | holdout 0.98946 | +0.00011 [0.00001, 0.00021] over `s15` |

| `s17` | base `v7` + e5-base cross-encoder rescored on `v7`'s bands (average precision inside the band 0.976 against 0.918) + e5-small as `xs2` + refined competition (`--xcons`) | holdout **0.99025**, **portal 0.981** | +0.00039 [0.00026, 0.00053] over `s14`, +0.00079 over `s16`; 4.74 candidates per S1; current best |

In flight (26 Sep 13:00): see `EXPERIMENTS.md` section 6.2 (`v8`, `v9`/`s21`, `s18`, `s19`, `s20`, `s22`). Portal: `s12` 0.971976, `s17` 0.981.

Options in code: `stack build --decoy --extra --xenc --sub-q N --tag T`, `stack train --set depth=...`, a stacked base model for a second consensus round; details in section 9.

## 5. Component reference

### 5.1 Text preparation (`src/ber/text.py`, `stages/prepare.py`)
Per record: HTML unescape, Latin-only accent strip, digit-for-letter fix (`C0mpany`), "doing business as" and domain split (`name1`, `name2`, `alias`), legal-form extraction (`Pvt Ltd` and `Private Limited` map to one form, kept as a feature so conflicting forms count against a pair), `core1` (name without legal form), romanised copies of name and address (`core_rom`, `addr_rom`, anyascii), a non-Latin fraction per field, and `norm_address(raw, country)` with France-only abbreviation and filler tables. Throughput about 10 microseconds per row (ASCII fast path). Records are addressed by `q` (S1 row id) and `pid = source * 10,000,000 + row id` for the pool.

### 5.2 Token blocking (`stages/block.py`) and the pruner (`stages/prune.py`)
A DuckDB table of tagged tokens for the 10.3M pool records with document frequencies. A pool record's blocking score for an S1 is the sum of IDF over shared tokens; tokens with df above 800 are ignored (raising the cap does not help and is very slow); token types: `n` name words, `a` address words and numbers, `p` 5-character prefixes, `c` a rare name word with a rare address word, `m` two rare name words, `d` two rare address words, `h` house number with a rare address word. Measured limit: 0.01% of true pairs share no token, so misses are ranking and truncation, not vocabulary. The cascade keeps K=100 raw candidates and a small XGBoost on S1-local features (score, shared tokens, per-type scores, rank, gap) keeps the best 30 (`p_block`, never used as a model feature). The pool index is cached and keyed by `INDEX_VERSION` (now 4; bump when token generation changes).

### 5.3 Dense channels (`stages/dense.py`, `stages/dense_all.py`)
Encoder `intfloat/multilingual-e5-small` (MIT, 118M parameters, 384-d, weights downloaded once, no data lookups), fp16, "query: " prefix, mean pooling. Fine-tuning is symmetric InfoNCE with in-batch negatives, one true pair per S1 per epoch, 3 epochs; `dense_all` adds one mined hard negative per S1 (a non-true candidate) and uses 250k S1 of all countries (756 s on an A10G), `dense` used 92k India S1. Retrieval is a GPU matrix product from each pool record to all S1 embeddings (no FAISS needed: 12.5M records encode in 26 minutes). The `zn2` report on the holdout chose the merge setting: top-1 with no cosine cut-off recovers 66% of the never-proposed pairs (16,025 of 24,230) for 1.09M extra pairs; top-3 (19.6M pairs) and top-5 (39M) add almost nothing; any cosine cut-off discards most of the gain. Merge is idempotent (backups `blocks/{split}_prededense` and `_predall`). New feature columns: `emb_cos`, `emb_rank` (names), `dall_cos`, `dall_rank`.

### 5.4 First-stage model (`stages/pairs.py`, `train_gpu.py`, `score_rest.py`)
`pairs.py` builds the features in chunks (rapidfuzz `cdist`, vectorised polars): 63 base features plus the 4 dense columns. `train_gpu.py` fits XGBoost on CUDA (about 850k positives in 8M pairs, aucpr about 0.998, 5 folds grouped by S1, threshold tuned OOF with exclusive assignment). `score_rest.py` scores the train S1 outside the sample with the final model (unbiased probabilities for stage two) and writes the holdout report. `predict.py` scores the test pairs part by part and holds `emit`, which applies the decision rule, writes both TSVs and runs the official validator.

### 5.5 Stack (`stages/stack.py`)
`build` writes chunked feature files under `WORK/stack[tag]/{split}`: `load_p1` (test: `output/<base>/pair_p.parquet`; train: `oof.parquet` plus `p1_rest/`; or, for a stacked base, its `oof_tune` and `holdout_pred`), `shortlist`, `pid_features` (record-level competition), `q_features` (S1-level), `digit_features`, `consensus_text_features`, optionally `decoy_features`, then `tfidf` adds `tf_name_cos` and `tf_addr_cos` in a bounded-memory pass. `train` uses a preallocated float32 matrix and QuantileDMatrix (an earlier version ran out of memory on the 15 GB notebook), depth 7, eta 0.08, 5 grouped folds, threshold tuned on the out-of-fold subsample, then a paired comparison on the holdout. `predict` scores the shortlist and calls `emit`.

### 5.6 Decision (`decision.py`)
Exclusive assignment (each pool record keeps its highest-probability S1, ties to the lower id), one global threshold, vectorised macro F0.5, bootstrap and paired bootstrap helpers. The threshold curve is flat between 0.6 and 0.7.

### 5.7 Analysis and validation scripts (`src/scripts/`)
`error_analysis.py` (loss decomposition, false positives and negatives, per-country), `miss_analysis.py` (never-proposed pairs by cause), `noise_analysis.py` (recall and precision by noise type), `country_expected.py` (F0.5 estimated from probabilities per country, a France proxy), `shortlist_eval.py` (cost of candidate cut-offs), `paired_models.py` (paired bootstrap between two stacked models), `check_submission.py` (all format and content rules of the statement), `qa_prepare.py`.

### 5.8 Cross-encoder (`stages/xenc.py`)
Commands (`data` in the `ber` env, `train` and `score` in the `pytorch` env): `xenc data --base v7 --dir xenc_v7` builds the band pairs and the fit set, `xenc train` fine-tunes (`--base-model`, `--model-dir`, `--set fit_more=...,band_lo=...,epochs=...`), `xenc score --split train|test --dir ...` writes `WORK/<dir>/<split>_xs.parquet`, `stack build --xenc --xenc-dir <dir>` joins it. The fit S1 come from `xenc_train_q` (outside the stage-1 sample, the holdout and the dense-encoder fitting sets; `fit_more` extends the set while keeping the first 400k), and `stack build` excludes them from stage two. Measured: average precision inside the band 0.969 (cross-encoder) against 0.897 (first stage) on train pairs (optimistic, it includes fit S1); the honest measure is the holdout stack, +0.0053.

## 6. Repository map

`code/business_entity_resolution/`: `configs/params.yaml` (sections prepare, sample, block, block_eval, train_gpu, prune, stack, expf, dense, dense_all), `src/ber/` (`config.py`, `text.py`, `split.py`, `decision.py`, `tracking.py`, `stages/*.py`), `src/scripts/`, `src/tests/` (28 tests, CPU only: blocking, prepare, prune, text, v0 end to end including stacking, the expected-F0.5 dynamic program checked against brute force, dense top-k and both merges, shortlist, decoy features), `Makefile` (base stages and `reproduce`), `README.md`, `requirements.txt` (pinned). `aws/notebook/` (lifecycle `onstart.sh`, `bootstrap.sh`, two-lane `jobrunner.sh`), `iam/` (policy JSONs). Every stage logs to `runs.jsonl` and MLflow (sqlite under `WORK/mlflow.db`) with the git commit.

Working data on the notebook (`/home/ec2-user/SageMaker/work`, about 70 GB of 98 GB): `parquet/`, `blocks/{split}_raw`, `blocks/{split}` plus backups, `dense/`, `dense_all/`, `features/{train,train_rest,test}`, `stack[tag]/`, `models/<name>/` (`xgb.json`, `config.json`, `holdout.json`, `oof*.parquet`, `p1_rest/`, `holdout_pred.parquet`), `output/<name>/`. Published results: `s3://sagemaker-us-east-1-567503593043/runs/<name>/`.

## 7. Commands that reproduce the current models (as they were run)

Common environment: `BER_DATA` (dataset folder with `train/` and `test/`), `BER_WORK`, `PYTHONPATH=src`. Stages in order:

1. `prepare`, `sample` (`python -m ber.stages.prepare`, `sample`).
2. Raw blocking: `block --split test --k 100 --out-name test_raw`, `block --split train --all-train --k 100 --out-name train_raw`; `prune --train`, `prune --apply train test`.
3. Name dense channel: `dense finetune`, `dense embed --split train|test`, `dense retrieve --split train|test`, `dense merge --split train|test`.
4. Name+address channel: `dense_all finetune`, `dense_all embed --split train|test`, `dense_all retrieve --split train|test`, `dense_all report` (decision), `dense_all merge --split test|train` (`k_merge 1`, `tau 0`, `k_merge_empty 3` only for the dropped `v6`; for `v5` `k_merge_empty` did not exist).
5. First stage: `pairs --split train`, `train_gpu --name v5`, `pairs --split train --rest`, `score_rest --name v5`, `pairs --split test`, `predict --name v5`.
6. Stack: `stack build --split train --base v5`, `stack tfidf --split train`, the same for `test`, `stack train --name s5 --base v5`, `stack predict --name s5` (`s6` is the same with `stack.shortlist_k 10`, `shortlist_pmin 0.005`, which are the current defaults in `params.yaml`; `s5` was built without them).
7. Checks: `python src/scripts/check_submission.py OUT_DIR TEST_DIR`, the official `validate_submission.py --check-ids`.

The dense stages need the notebook `pytorch` conda env (torch plus transformers); everything else runs in the `ber` env (Python 3.12, `requirements.txt`).

## 8. Experiments that did not help (kept so nobody repeats them)

| Experiment | Result | Reference |
|---|---|---|
| Extra token types `g`, `x`, `k` in the blocker | no recall gain, glued names displaced better candidates; reverted | `research.md` section 14 |
| Higher df cap (5000, 20000) | recall 0.9403 and 0.9412 against 0.9416, 13 to 160 times slower | `handoff.md` section 7 |
| Calibration plus expected-F0.5 selection | +0.0001, interval includes 0 | `research.md` section 17 |
| Name-only dense top-10 instead of top-5 | about +1,000 pairs (0.2%) | `research.md` section 19 |
| `dense_all` top-3 or top-5, or with a cosine cut-off | 18 to 38M more pairs for +1.5 to 2.3k pairs; the cut-off discards most of the gain | `zn2` report |
| Extra neighbours for empty-address pool records (`v6`, `s7`) | 0.97557 and 0.98309, no change | 26 Sep run |
| Stack ablation: name/address consensus features | +0.0001 (kept, harmless) | `s3nocons` |
| Kaggle-style advice: XGBoost/CatBoost blend | not tried; the Foursquare 4th place team also found stacking gains small | `research.md` section 18.1 |

## 9. What is next (backlog, expected gains, status)

See `TEAM_GUIDE.md` section 4 for owners. Current state on 26 Sep 02:15: the notebook is being moved to `ml.g5.16xlarge` (64 vCPU, 256 GB, one A10G, $5.12/h, budget 270 credits) with two parallel job lanes; the first experiments queued for it are `ra1` (`s8`: decoy edit features plus extra carried columns, and a variant without the decoy features) and `rb1` (`s9`: second consensus round on `s6`). After those: state-name normalisation (`OK` and `Oklahoma`, Indic states; needs a re-prepare and rebuild), cross-encoder scores for the uncertain band from a teammate (added as one stack feature), a seed ensemble and larger training samples, the 5 S2 + 6 S3 cap, a threshold check on the portal, and the freeze (rebuild the code zip, one clean reproduction, validator with `--check-ids`, methodology document).

## 12. Where the remaining loss is, and the holdout to portal gap (26 Sep 12:10 IST)

`s15` on the locked holdout: oracle on the candidates 0.9957, blocking loss 0.0043, matcher loss 0.0064, precision 0.9980, recall against all true pairs 0.9718 (candidates contain 98.56%). US 0.98968, India 0.98886. Missed true pairs by cause: pool address empty 10,833 of about 14.6k (74%; 22,854 true pairs, 75% reach the candidates, 53% matched), alias-only names 1,719, words injected or dropped 2,720, typos 2,169, house number differs 635. False positives 1,033 (58% distractors without an owner). The empty-address records carry only a name, so much of this is ambiguity; the theoretical best threshold for F0.5 (F* / 1.25 = about 0.79) is close to the tuned 0.71.

The portal read `s12` 0.9720 against a holdout of 0.9851 (gap 0.0131; `s4` 0.0178, `v2` 0.0125). Estimated causes: the test is harder than the training data for every country (pool records per S1 5.8 against 4.7, unowned pool share about 40% against 26%; model-based estimate about -0.003), the country mix (-0.001 now that India is close to the US), France (model-based estimate 0.9866 against 0.9940 for the US, -0.001), public-subset noise (about 0.002), leaving about 0.008 unexplained: France probabilities not calibrated (no French labels), or covariate shift. Count-based first-stage features depend on the set size (the test has 21% fewer S1 than the train), which makes names look rarer on the test; `pairs.joint_counts` counts over train and test together, and `src/scripts/adv_validation.py` measures the shift directly. Tools for the remaining diagnosis: per-country thresholds (`reemit.py`), per-country probe files (empty one country's predictions; the score drop gives its F0.5), and stack training weights that make the training pairs look like the test.

## 10. Risks and open questions
- **Holdout to portal gap.** 0.012 to 0.018 so far, partly explained by mix and distractor density (about 0.008), the rest unexplained (France miscalibration, public-subset noise). `s5` and `s6` portal scores will show whether the dense channels narrowed it.
- **Feature shift.** `dall_cos` is the strongest new feature and its test distribution differs (top-1 cosine 0.795 on test against 0.825 on train, more unowned pool records). A fallback without it is `s4`.
- **France** has no labels; the model estimates it near the US (0.9786 against 0.9851), which is not a measurement.
- **Ambiguity floor.** Empty-address pool records (4.4% of true pairs, 50% matched) and identical-name look-alikes are undecidable from the data.
- **Compute.** One GPU notebook, sequential queue; deadline Sun 27 Sep 23:59 IST; 5 portal submissions per day.

## 11. Teammate plans reviewed (`research.md` section 19.6)
`docs/archive/v1.md`, `docs/archive/v2.md` (record-centric layers): integrated where useful (consensus features). `v4` plan (cross-encoder on the uncertain band, dense blocking with an English-only encoder): the dense blocking part was already done and its encoder cannot read Indic scripts; the cross-encoder with hard negatives and `[NUM]`/`[POSTCODE]` tags stays the teammate's own experiment. `v5` branch on `sai` at `1688d18` (state names, exact keys, sibling expansion, cross-encoder, decoy edit features, per-country thresholds, cap): decoy features, state names and the cap adopted into our backlog; the model names `v5` and `s5` collide with ours on a shared notebook, so they must use their own names and code prefix. "Executive Summary.pdf" (learned per-S1 truncation, hard-negative clustering): the truncation idea is folded into the shortlist work; its cost estimates and "French recall audit" are not usable (no French labels).
