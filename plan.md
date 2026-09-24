# Plan v1: baseline, evaluation and MLOps (approved 2026-09-25)

Written 2026-09-25. Supersedes the informal plan in `research.md` section 4 and section 10. Facts about the data are in `context.md` section 2b. Nothing below is implemented yet except v0 (`code/business_entity_resolution/`), which this plan replaces at scale.

## Status (2026-09-25 about 02:00 IST)

- Phase 0 (foundation): done. Normaliser, `prepare`, `sample`, Makefile with hash stamps, MLflow logging, live job logs, tests.
- Phase 1 (blocking): in progress. First measurement with the plan's channels A (`n`, `a`, `p`) and composite keys: recall 0.85 at 29 candidates per S1, gate 1 not met. Root cause is the document-frequency cap (800) being far too small for a 10.3M-record pool. Composite keys (rare name word with rare address word) were the strongest channel (0.78 alone), so more composite types were added (`m` name pair, `d` address pair, `h` house number with address word). The dense multilingual channel (E) is still open and will be added only if the lexical reachability diagnosis shows true pairs that share no token.
- Phases 2 to 4: not started.
- One design change since approval: the pool-side token index is cached in a persistent DuckDB file keyed by an index-parameter hash, so `cap_df`, per-type limits and K can be tuned in seconds instead of rebuilding for about 3 minutes.

## 0. Review of the earlier plan: what holds and what changes

| Earlier idea | Verdict | Change |
|---|---|---|
| Rare-token inverted index for blocking, dense embeddings only for non-Latin | Keep, but a single channel is not enough | Union of several cheap channels, each measured separately on train. Names can be completely different (alias/DBA) and digits in addresses can be altered, so a name-only or exact-address-only channel will miss real matches |
| Train on a sample of 250k S1 | Keep, but the sampling rule was wrong as stated | **Sample the queries (S1) and keep the full S2/S3 pool.** Sampling the pool too would make distractors 10x sparser than in test and make precision look better than it is |
| Global one-to-one assignment | Keep, and strengthen | Ownership is exclusive on train (0 of 7.64M matched ids claimed twice). Use it as a decision step, and use competitor statistics as model features |
| One global threshold tuned on OOF | Replace | Per-entity expected-F0.5 set selection on calibrated probabilities (section 5). The metric is macro-averaged per S1 entity, so choosing the best prefix of each entity's ranked candidates beats one threshold |
| LightGBM first, cross-encoder later | Keep the order | Cross-encoder only for the ambiguous band, and only after the baseline has a leaderboard score |
| Address equality is a strong signal | Dangerous | True in train (S1 address almost unique, max 14 repeats) but false in France (same address up to 101 times in test S1). The model must see how many entities compete for one address, and validation must include a collision stress set |
| Python per-pair features | Not viable at 35M to 65M pairs on 4 vCPU | Vectorised features only, streamed in shards, no per-pair Python loops |
| MLflow, notes in `runs.jsonl` | Keep, add stage caching | DVC-style DAG with hashed inputs so a 3-day run can be re-executed from scratch and partly reused (section 7) |

## 1. Goal and success criteria

- Baseline v1 = the simplest pipeline that is defensible as the first serious submission: multi-channel blocking, feature-based LightGBM, calibrated probabilities, expected-F0.5 decision with exclusive assignment. No neural model needed for the baseline.
- Success gates (each is a go/no-go before spending more compute):
  1. Blocking: pair recall at least 0.98 on the train sample at no more than 30 candidates per S1, and reported separately for US and India, singletons excluded.
  2. Matcher: out-of-fold macro F0.5 on a held-out S1 sample, reported per country, per singleton/matched, and on the collision stress set.
  3. First valid submission within roughly 16 hours of the start (ties go to the earlier submitter).
- Final quality target is not fixed yet; it will be set after gate 2 gives the first honest number.

## 2. Data layout and sampling protocol

- Convert the seven TSVs to Parquet once on the g5 (`work/parquet/{train,test}/source{1,2,3}.parquet`), with an integer `rid` per record and normalised text columns. Keep the raw TSVs read-only.
- Text normalisation (deterministic, no lookups): `html.unescape`, Latin-only accent stripping (already fixed), leetspeak repair only in tokens that mix letters and digits (`c0mpany`), domain-style names split into words (`ipower.com`), `doing business as` / `dba` split into two names, glued city suffix cleanup (`chicagocdp`), abbreviation expansion, legal-form words kept as a separate field rather than dropped.
- Sampling for training (query sampling, full pool):
  - Pick a random 250k of the 2.2M train S1 records (stratified by country and by number of matches).
  - Blocking runs for all S1 of both splits (retrieval is cheap and is needed on test anyway). Expensive pair features are computed only for the sampled S1's candidates.
  - Candidate-side statistics (how many S1 entities compete for one S2/S3 record, and how strong the best competitor is) are taken from the blocking scores of all S1, so they match test-time conditions.
- Folds: grouped K-fold by S1 (5 folds). Owner exclusivity means no record's label leaks across folds through a different S1.
- Three fixed validation views on the OOF predictions: overall, per country, and a **collision set** (S1 entities whose normalised address is shared with at least one other S1). For France-like behaviour, an additional synthetic stress: duplicate addresses across S1 entities in the sample and check precision.
- Country generalisation check: train on US only, evaluate India, and the reverse (a proxy for the unseen France).

## 3. Blocking design

Channels (all run per S1 against the full S2+S3 pool, results unioned and deduplicated):

| Channel | Idea | Guards |
|---|---|---|
| A. Rare-token overlap | Each record's tokens (name and address, Latin normalised, plus 5-character prefixes to survive suffix corruption). Weighted overlap by IDF, top-K per S1 | Drop tokens with document frequency above a cap (about 2,000 records); cap candidates per token |
| B. Address key | House number + street tokens, and city + street tokens | Cap block size; for oversized blocks (France) require a name token in the key too |
| C. Name key | Rare name token + city token; name prefix + first address token | Same caps |
| D. Char n-gram TF-IDF sparse kNN | Catches typos and leetspeak channels A to C miss | Run only on the records that channels A to C left with fewer than K candidates, or on a sample to measure marginal gain |
| E. Dense multilingual embedding (name only) | Bridges Devanagari, Telugu and Malayalam names to Latin ones. About 1M non-Latin records, fits the A10G budget | Only for non-Latin names; model chosen by the throughput and recall benchmark below |

- Engine: DuckDB (MIT) for the token joins, out-of-core on the g5's 15 GB; polars for column ops. Embeddings on the GPU.
- Each channel's marginal recall is measured on the train sample so channels that do not pay for themselves are dropped. Candidates are ranked by a cheap combined score and truncated at K before the matcher.
- Output of the stage is the exact `candidate_pairs` set that the matcher sees, which is also what the submission requires.

## 4. Matcher (stage 1: LightGBM)

- Features (all vectorised, computed in shards of about 2 to 3M pairs and streamed to Parquet):
  - Name: rapidfuzz ratios and token variants, Jaro-Winkler, normalised Levenshtein, TF-IDF cosines (word and char), acronym match, legal-form agreement, first-token equality, length and token-count ratios, domain-name match, alias (DBA) match.
  - Address: the same string measures, plus number features (house-number agreement, digit edit distance, PIN/postal code equality or conflict), token overlap after abbreviation expansion, missing-field flags.
  - Competition: rank and gap of the pair within its S1's candidate list, and within the candidate record's list of claimants (from all-S1 blocking scores), address multiplicity (how many S1 share this address), name genericness (name frequency in S1 and in the pool).
  - Source flag (S2 vs S3). No country one-hot; a numeric "script mix" feature instead so France and non-Latin text do not need a label.
- Training pairs: candidates of the sampled S1, labelled from ground truth. Hard negatives come free from blocking (top-ranked non-matches), which is exactly what the model faces at test time.
- Calibration: isotonic regression on OOF predictions, fit separately for S2 and S3 pairs; check calibration per country.
- Two models are trained with different seeds/features and blended only if OOF shows a gain (keeps the baseline simple to explain).

## 5. Decision rule (the part that maps to the metric)

1. **Exclusive assignment.** Every S2/S3 record is offered only to the claimant S1 with the highest calibrated probability (ties broken by rank); the others lose that candidate.
2. **Per-entity expected-F0.5 selection.** For each S1, sort remaining candidates by probability and choose the prefix length k (including k = 0) that maximises the expected F0.5 under the calibrated probabilities. Empty prediction has expected score equal to the probability the entity is a singleton, so singletons are handled by the same rule. Compute exactly with a small dynamic programme over at most K candidates, vectorised in numpy.
3. **Veto rules for precision** (small, auditable): conflicting PIN/postal code, conflicting non-empty house number when both sides are complete, country mismatch.
4. Tune a single global "conservatism" scalar (a multiplier on precision weight) on the OOF collision set so that France-like crowds do not over-merge. Reported in the run log.

## 6. Stage 2 (after the baseline scores): reranker for the ambiguous band

- Ambiguous band = candidates where stage-1 calibrated probability is in roughly 0.2 to 0.8, or where two S1 entities compete for the same record. Expected volume is a few percent of candidate pairs, on the order of 1 to 5M pairs at inference.
- Model: cross-encoder over `name | address` pair text from a multilingual MIT-licensed base (xlm-roberta-base or mdeberta-v3-base, about 280M), fine-tuned with binary cross-entropy on hard negatives from blocking. Alternative starting point: Qwen3-Reranker-0.6B (Apache-2.0).
- Feed the cross-encoder score back as a feature into a second LightGBM (stacking), then reuse the same decision rule.
- Before committing: benchmark inference throughput and fine-tuning time on the A10G with a 100k-pair sample; go only if the ambiguous band is inferable in under about 60 minutes.
- LLM adjudication (Qwen3-8B, Apache-2.0) only for the residual doubt set, and only if there is time; it sits at the parameter cap and is slow.

## 7. MLOps design

Constraints: one 4 vCPU / 15 GB g5 notebook, S3 job queue instead of SSH, 72 hours, 5 submissions per day.

- **Repository layout** (under `code/business_entity_resolution/`): `src/ber/` (library), `configs/` (`params.yaml` with every tunable, one file per experiment override), `stages/` (thin CLI wrappers: `prepare`, `sample`, `block`, `features`, `train`, `calibrate`, `decide`, `predict`), `tests/`, `requirements.txt` (pinned), `README.md`.
- **Pipeline as a DAG with stage caching (decision: Makefile + MLflow, no DVC).** A `Makefile` declares each stage's inputs, params file and outputs; a stage writes a `.done` marker containing the hash of its inputs, params and code, and is skipped when the hash is unchanged. So a rerun after a threshold change recomputes only `decide` and `predict`. Stage outputs are synced to the S3 run directory. The stage boundaries (`prepare`, `sample`, `block`, `features`, `train`, `calibrate`, `decide`, `predict`) are unchanged from the DVC design, so DVC can be adopted later without restructuring.
- **Experiment tracking.** MLflow (Apache-2.0) with a file/sqlite backend on the notebook, artifacts mirrored to S3. Every run logs: git SHA of the synced code, data version (hash of the Parquet inputs), the full params, blocking recall per channel, OOF macro F0.5 overall/per country/collision set, calibration error, decision settings, runtime per stage, and the leaderboard score when a submission is made. This is the submission version history the guidelines ask for.
- **Artifacts** under `s3://sagemaker-us-east-1-567503593043/runs/<run_id>/`: config snapshot, models, OOF predictions, candidate set, both output TSVs, validation report. `run_id` = timestamp + short git SHA.
- **Determinism.** Fixed seeds everywhere, pinned versions, single entry point `python -m ber.pipeline --config configs/<name>.yaml`; test-time output regenerated from the frozen model directory only.
- **Sharding and resumability.** Every heavy stage writes shard files with completion markers; a crash resumes at the last shard rather than restarting.
- **Compute orchestration.** Jobs go through the existing S3 queue; a small extension adds live log streaming to S3 every 30 s and a status file so progress is visible from the laptop without opening Jupyter. Idle auto-stop stays job-aware.
- **Quality gates in CI-style checks.** Unit tests for normalisation, metric, decision DP and validator; a smoke run on a 1% sample must pass before any full run; the official `utils/validate_submission.py` must print PASS before any upload.
- **Model registry.** Not a service: a run directory plus a `promoted.json` pointer to the run behind the current best submission.
- **Licences.** Every pretrained model and library is logged with its licence in the run; only MIT/Apache-2.0 models, at most 8B parameters.
- **No external lookups.** Enforced by design: no network calls in stages after the one-time model weight download (weights cached under `work/models/`); tests fail if a stage imports an HTTP client.

## 8. Schedule and gates (window ends Sun 27 Sep 23:59 IST)

| Phase | When | Deliverable | Gate |
|---|---|---|---|
| 0 Foundation | Fri 25 Sep, 0 to 4 h | Parquet, normaliser, sampling, DVC/MLflow scaffold, smoke tests | Smoke run green |
| 1 Blocking | 4 to 10 h | Channels A to C measured on train sample, then D/E as needed; K chosen | Recall at least 0.98 at no more than 30 candidates per S1 |
| 2 Matcher and decision | 10 to 18 h | Vectorised features, LightGBM, calibration, decision rule, OOF report, full test inference | First honest OOF; first valid submission (submission 1) |
| 3 Improvements | Sat 26 Sep | Ablations one at a time: exclusive assignment, expected-F0.5, embeddings for non-Latin, collision features; cross-encoder benchmark and, if it pays, stage 2 | Each change must improve OOF and the collision set before a submission is spent on it |
| 4 Hardening | Sun 27 Sep to 20:00 | Freeze, full reproduction from scratch on a clean env, methodology doc, zip, official validator PASS | Reproduced outputs identical to the promoted run |
| Buffer | last 4 h | Final submission and package check | none |

Submission policy: at most 3 uploads per day on distinct hypotheses, never to tune thresholds against the public leaderboard (it uses a subset of the test set); keep 2 per day in reserve.

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| France crowds (repeated addresses, generic names) break address-driven scoring | Collision features, collision stress set, a conservatism scalar tuned on it, veto rules, expected-F0.5 decision |
| Blocking misses alias-only matches | Measure the fraction of true pairs found by no channel; if material, add dense name channel or address-only channel for those |
| 4 vCPU too slow for full-scale features | Sharded vectorised features, DuckDB for joins, measure throughput on the sample first, reduce K or drop the slowest feature before scaling |
| Calibration drifts on France | Country-wise calibration check on India vs US as proxy; conservative scalar; keep stage 1 monotone and simple |
| Leaderboard overfitting on the public subset | Decisions come from OOF, not leaderboard; limit submissions |
| Notebook stops or session dies | Shard markers, S3 artifacts, job-aware auto-stop, no state only on the instance |
| Cost | About $1.01/h while the g5 is up; stop it during long analysis on the laptop; budget alert already exists on the account |

## 10. Decisions needed from the team before implementation

Decided on 2026-09-25:
1. MLOps depth: **Makefile + MLflow** (no DVC).
2. Team: **single account only** (`test-notebook` in account A); no parallel teammate tracks for now. Revisit after the first submission if compute becomes the bottleneck.
3. Day-1 scope: **baseline only** (blocking, LightGBM, calibration, expected-F0.5 decision, exclusive assignment). Cross-encoder is decided after the first real score and the ambiguous-band size.

Still open, settled by measurement rather than opinion:
4. Embedding model for the non-Latin channel (bge-m3 MIT vs multilingual-e5-base MIT), chosen by the throughput and recall benchmark in phase 1.

## 11. References used for this revision

- Decision-theoretic F-measure optimisation: Nan Ye et al., "Optimizing F-measure: A Tale of Two Approaches" (arXiv 1206.4625); "Optimal Decision-Theoretic Classification Using Non-Decomposable Performance Metrics" (arXiv 1505.01802); Lipton et al., "Thresholding Classifiers to Maximize F1 Score" (arXiv 1402.1892).
- Address matching with bi-encoder plus cross-encoder rerank: arXiv 2307.02300; ensemble address matching on DuckDB: PMC13426768.
- Hard negatives from blocking output: Block-SCL (arXiv 2207.02008); SC-Block (arXiv 2303.03132); retrieval hard-negative mining (arXiv 2411.02404).
- Blocking scale and block-size guards: Splink docs and issue 1448 (array-intersection blocking as a hash join); GoldenMatch scale envelope (polars below 500K rows, DuckDB for 500K to 50M, block-size and cardinality guards).
- Embedding throughput reference: RunPod bge-m3 benchmark (about 1,345 texts/s on an H200 at 512-token passages; our texts are about 20 tokens so throughput is much higher, to be measured on the A10G).
- MLOps: DVC pipelines with S3 remotes plus MLflow tracking (RunPod and DVC docs); SageMaker checkpointing and managed spot (AWS docs).
