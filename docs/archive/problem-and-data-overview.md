# Understanding: problem, dataset, current approach

Written 2026-09-25. One-file summary of the Amazon ML Challenge 2026 task as understood from:
- `ml-challenge.pdf`: the task specification
- `prob_statement.pdf`: the general challenge rules
- `context.md`, `research.md`, `plan.md`
- the `ber` code in `code/business_entity_resolution/`

**Source tags** used below:
- **[spec]**: from the problem statement
- **[peek]**: teammates' look at file headers
- **[stats]**: figures quoted in `plan.md` from the stats job
- **[est]**: my own arithmetic from file sizes

I could not open the data myself: this laptop's AWS credentials get Access Denied on both buckets. Anything not tagged [spec] should be confirmed with the checks in section 9.

---

## 1. The problem in one paragraph

Three lists describe the same real-world businesses, each with a name, address and country.
- **Source 1 (S1)** is the clean reference list, with each business exactly once.
- **Source 2 (S2)** and **Source 3 (S3)** are noisy: typos, abbreviations, reordered words, other scripts, missing address parts, trade names.
- No ID links the lists. For **every S1 business**, we must list the S2/S3 records that refer to the same business. That can be zero, one or many.

This is **entity resolution** with a fixed reference side. It's a "retrieve then verify" problem per S1 record, not clustering of the whole graph. [spec]

## 2. Inputs and outputs [spec]

### 2.1 Files
All files are tab-separated. Always read them with `sep="\t"`, because addresses and ID lists contain commas.

| Split | Files |
|---|---|
| train | `train_source1.tsv`, `train_source2.tsv`, `train_source3.tsv`, `train_ground_truth.tsv` |
| test | `test_source1.tsv`, `test_source2.tsv`, `test_source3.tsv` (no labels) |
| helpers | `utils/validate_submission.py` (stdlib-only format checker), `docs/Documentation_template.md` |

### 2.2 Record schema (all three sources)
| Column | Meaning |
|---|---|
| `entity_id` | Unique ID. The prefix gives the source: `S1-`, `S2-`, `S3-` |
| `business_name` | Name. It can have abbreviations, legal-suffix differences, typos, transliterations and trade names |
| `business_address` | Address. It can be partial, reordered or landmark-based ("Near SBI ATM"), and can be missing its PIN or state |
| `country` | An **open set of labels**. Train has `US` and `India`; **test also has `France`**, which train never contains |

There is no separate "source" column: the ID prefix and the file name tell you the source.

### 2.3 Ground truth (train only)
- `source1_entity_id`: an S1 ID.
- `matched_entity_ids`: comma-separated S2/S3 IDs. It is **empty for singletons** (S1 records with no match).

### 2.4 What we must produce
| File | Content | Scored? |
|---|---|---|
| `output/matching_results.tsv` | `source1_entity_id`, `matched_entity_ids`: the final matches | **Yes**, the only scored file |
| `output/candidate_pairs.tsv` | `source1_entity_id`, `candidate_entity_ids`: the exact shortlist the matching model scored | No. It's audited for blocking recall and reduction ratio |

Rules for both files. Breaking any of these gets the file **rejected**:
- exactly one row per test S1 ID
- the list is empty when there's nothing to report
- only S2/S3 IDs that exist in the test set
- no self-matches to S1
- no duplicate IDs in a list and no duplicate S1 rows
- every matched ID must also appear in the candidate list, or the validator warns about a pipeline bug

### 2.5 The final package [spec]
```
<team>_submission.zip
├── output/{matching_results.tsv, candidate_pairs.tsv}
├── code/business_entity_resolution/{src/, README.md, requirements.txt}
└── Documentation_template.md   (filled in)
```

The write-up must cover:
- the methodology
- candidate generation / blocking
- model architecture and feature engineering
- experiments and conclusions
- anything else relevant

`prob_statement.pdf` asks for 1–2 pages. The code needs comments on its functions. Top-10 packages are reviewed in detail.

### 2.6 Rules and limits [spec]
- **Models:** the final model must be MIT or Apache-2.0 licensed, **at most 8B parameters**.
- **No external data:** no ER APIs, registries, geocoding or internet augmentation. Breaking this means **disqualification**. Pretrained weights of allowed models plus the provided data are fine. Document every model and its licence.
- **Window:** 25 Sep 2026 00:00 IST to 27 Sep 23:59 IST.
- **Submissions:** at most **5 per day**, and the version history of every submission must be kept.
- **Leaderboards:** the public one scores a subset of the test set, the private one the rest. Shortlisting considers both. Ties go to the earlier submission.
- **Logins:** one device per participant; simultaneous logins can terminate the attempt.

---

## 3. The metric, and what it means for decisions

### 3.1 Definition [spec]
For each S1 entity, with P = precision and R = recall of its predicted list:

    F0.5 = 1.25 · P · R / (0.25 · P + R)

The final score is the **plain average over all S1 entities**, singletons included.
- A true singleton scores **1.0 if we predict nothing** and **0.0 if we predict anything**.

### 3.2 Worked values
| True matches | We predict | P | R | F0.5 |
|---|---|---|---|---|
| 0 (singleton) | nothing | – | – | **1.00** |
| 0 (singleton) | 1 wrong | 0 | – | **0.00** |
| 1 | the right one | 1 | 1 | 1.00 |
| 1 | right + 1 wrong | 0.5 | 1 | 0.56 |
| 1 | nothing | – | 0 | 0.00 |
| 2 | 1 right | 1 | 0.5 | 0.83 |
| 2 | 2 right + 1 wrong | 0.67 | 1 | 0.71 (the spec's example) |
| 2 | 1 right + 1 wrong | 0.5 | 0.5 | 0.50 |
| 4 | 1 right | 1 | 0.25 | 0.63 |
| 4 | 2 right | 1 | 0.5 | 0.83 |
| 4 | 3 right | 1 | 0.75 | 0.94 |
| 4 | 4 right + 1 wrong | 0.8 | 1 | 0.83 |

### 3.3 Consequences
1. **Singletons are all or nothing.** Deciding whether an S1 has any match is its own decision and deserves its own model.
2. **Partial recall is paid well.** Two sure matches out of four already score 0.83, so drop doubtful candidates rather than add them.
3. **Each wrong ID costs about as much as a missing one** when the list is otherwise right (0.83 either way for n = 4). On a singleton, one wrong ID costs the whole entity.
4. **Scoring per entity means one global threshold is suboptimal.** The best cutoff depends on how many matches that S1 likely has and how confident each candidate is. Choose per entity the prefix of its ranked candidates (including the empty list) with the highest expected F0.5.
5. **The metric can't reward a match the shortlist never produced.** Blocking recall is the ceiling.

---

## 4. The dataset

### 4.1 Location and size
Bucket: `s3://ml-challenge-nooglers/ml-challenge-2026/raw/v1/`. It is owned by a teammate's account and readable by account A and the notebook role. Sizes are [peek].

| File | Size | Rows [stats]/[est] |
|---|---|---|
| `train_source1.tsv` | ~200 MiB | **~2.2M** [stats] |
| `train_source2.tsv` | ~467 MiB | ~5M [est, at ~95 bytes/row like S1] |
| `train_source3.tsv` | ~480 MiB | ~5M [est] |
| `train_ground_truth.tsv` | ~121 MiB | one row per train S1; **7.64M matched IDs in total** [stats] |
| `test_source1.tsv` | ~167 MiB | ~1.8M [est] |
| `test_source2.tsv` | ~486 MiB | ~5M [est] |
| `test_source3.tsv` | ~483 MiB | ~5M [est] |
| `student_resource.zip` | ~1 GiB | original bundle |

In total there are roughly **2.9 GB of TSV and about 25M records**, about 12M per split.

### 4.2 Match structure
- **Matches per S1:** 7.64M matched IDs over 2.2M S1 gives **about 3.5 per S1 on average** [est]. That includes singletons, so matched S1s average more. Ground-truth rows often list **3 to 5** IDs across S2 and S3 [peek].
  - So one S1 often has **several records in the same source**: S2 and S3 contain internal duplicates.
- **Exclusive ownership:** **0 of the 7.64M** matched S2/S3 IDs is claimed by more than one S1 [stats]. Every S2/S3 record belongs to at most one S1, which enables a "give each record to its best S1" step.
- **Distractors:** if S2+S3 together hold about 10M train records and 7.64M are matched, then **about 25% of the pool matches nothing** [est]. The shortlist must hold these back.
- **Singleton share:** **unknown**. This is the most important number still missing, because it sets how cautious the decision must be.
- **Address uniqueness:** S1 addresses are almost unique in train (at most 14 S1s share one address), but **in test France, up to 101 S1s share one address** [stats]. These are likely shared business centres or registered-office addresses.
- **Non-Latin scripts:** about **1M records have non-Latin names** (Devanagari, Telugu, Malayalam) [stats].

### 4.3 What records look like [peek]
- **IDs** are random integers such as `S1-925783039`. There's no order or signal in them.
- **India:**
  - S2 names can be **Devanagari** (`राम मार्केटिंग प्राइवेट लिमिटेड` = "Ram Marketing Private Limited"), while addresses are Latin and upper-case (`KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi`).
  - There are Indian legal forms (Pvt Ltd) and landmark phrases (Near …, Opp …).
- **US:**
  - The order of address parts varies (`GREENSBORO, NC, 19 1/2 STARDUST TRAIL`).
  - **Some S3 names are web domains** (`wilfordhancock.com`).
  - **Some addresses are empty.**
- **France (test only):**
  - Abbreviated street types (`63 R. DE DIEPPE, LILLE, Hauts-de-France`, where R. = rue).
  - Names with French legal forms (`… Sarl`, `SCI …`).
- **Noise the spec says to expect:**
  - Name: Corp/Corporation, Pvt/Private, Ltd/Limited, legal-suffix differences, trade names, & vs "and", reordered words, typos.
  - Address: Rd/Road, St/Street, transliteration variants, missing PIN or state, landmark references, municipal numbering formats, reordered parts.
- **Other patterns `plan.md` notes to handle:**
  - leetspeak-like corruption (`c0mpany`)
  - names glued to city fragments
  - HTML entities
  - "doing business as" / "dba" pairs (two names in one field)

### 4.4 Why this data is hard
| Difficulty | Effect |
|---|---|
| ~12M records per split | Comparing everything with everything is impossible, so blocking must scale |
| Cross-script names | Character n-grams and edit distances give ~0 for `राम` vs `Ram`; we need romanisation or multilingual embeddings |
| Trade names and domains | The name can be entirely different, so the address must carry the match |
| Altered digits and missing parts | Exact address keys alone miss real matches |
| Internal duplicates (several matches per source) | The model must accept several near-identical candidates, not just the best one |
| ~25% distractors, plus singletons | The model must be able to answer "nothing" |
| France unseen, with heavily shared addresses | A model that trusts address equality will merge different businesses; trees can't extrapolate past the train range (≤ 14 sharers) |

---

## 5. Resources

| Resource | Details |
|---|---|
| **g5 notebook** `test-notebook` (account A) | ml.g5.xlarge: A10G 24 GB GPU, **4 vCPU, 16 GB RAM**, 100 GB disk, ~$1/h. Auto-stops after 1 h idle, but not while a job runs. `ber` conda env (Python 3.12). S3 **job queue**: put a `.sh` in `s3://sagemaker-us-east-1-567503593043/jobs/pending/`; the log appears in `jobs/done/` |
| AWS accounts | A: Paid plan, $200 credits. B, C, D: Free plan, $200 each. All us-east-1 |
| Modal | $30/month credit per workspace, up to 10 GPUs in parallel; large CPU containers available |
| Kaggle ×4 | T4×2 or P100, ~30 GB RAM, 30 GPU-hours/week each (resets Sat 00:00 UTC) |
| SSH helper (Sai's `remote-setup.md`) | Works, but lands in an old container (PyTorch 1.9, Python 3.8). Not the main route |
| Laptop | Code and tests only. The home network is slow, so never download bulk data here. **Currently has no access to account A's buckets** |

The real constraint is **CPU and RAM on the g5** (4 cores, 16 GB), not the GPU.

---

## 6. Current approach: baseline v0 (`code/business_entity_resolution/src/ber`)

Status: written and tested **only on synthetic data** (`synth.py`, 2 pytest tests). It has produced **no real score and no submission**.

| Step | Module | What it does |
|---|---|---|
| Load | `data.py` | Reads the TSVs into pandas and adds normalised columns: `name`, `core` (name without legal words), `addr`, `ctry`, `nums` (a set of digits), `both` = core + addr. Ground truth becomes a dict from S1 ID to a set of IDs |
| Normalise | `normalize.py` | Strips Latin accents only (Devanagari is kept intact after a fix), lowercases, removes punctuation, expands abbreviations (Corp→corporation, Rd→road…), drops legal words from US/India/France for `core`, maps country aliases |
| Block | `blocking.py` | Two TF-IDF character n-gram views (`core`, `both`). For each S1, the top 25 S2+S3 records by cosine in each view, **within the same country**, falling back to all countries if the country is unseen. Union of both views. `recall()` reports the recall ceiling and the reduction ratio |
| Features | `features.py` | About 49 per pair: rapidfuzz ratio, partial, token-sort and token-set on name and address; Jaro-Winkler and Levenshtein; token and digit Jaccard; PIN match and conflict; acronym match; first-token and exact-name equality; lengths; same country; `is_s3`; 4 TF-IDF cosines; each pair's **rank and gap** among the S1's candidates and among the candidate's S1s; candidate counts |
| Model | `model.py` | LightGBM binary classifier, **5-fold cross-validation grouped by S1** so an entity never appears in both train and validation. Out-of-fold (OOF) probabilities |
| Decide | `decide.py` | **One global threshold** (grid 0.20 to 0.94), optionally "one-to-one" (each S2/S3 kept only under its best S1), chosen by the best OOF macro F0.5 |
| Metric | `metrics.py` | Exact challenge metric, plus a singleton vs matched breakdown. Checked against the 0.714 example |
| Output | `run.py`, `validate.py` | `train` / `predict` commands; writes both TSVs and runs a local copy of the validator |

### 6.1 Known problems with v0
1. **It doesn't scale.**
   - Blocking turns sparse similarity into dense chunks, about 2.2M × 5M per country, which takes days.
   - Pandas holding Python sets per row for ~12M records, plus per-pair Python loops in the features, won't fit 16 GB and 4 cores.
2. **No cross-script matching.** About 1M non-Latin names get almost no recall or useful features.
3. **The decision rule doesn't fit the metric:** one global threshold, uncalibrated probabilities, no entity-level "has any match" decision.
4. **France precision risk:** address equality is learned as strong evidence where addresses are unique.
5. **Train/test feature shift:**
   - TF-IDF is refit per split.
   - Competition features (rank and count among a candidate's S1s) depend on which S1s are present, which differs when training on a sample.
6. **Evaluation can't show overfitting:**
   - one OOF number, with the threshold tuned on that same OOF
   - no holdout, no country-transfer test, no confidence intervals
7. **The GPU is unused** while the CPU is the bottleneck.

---

## 7. Planned direction

`plan.md` (by the team lead) defines "v1":
- Parquet data; sample S1 queries for training but keep the full S2/S3 pool.
- Multi-channel blocking: rare tokens, address keys, name keys, character TF-IDF, and dense embeddings for non-Latin names.
- Vectorised features, LightGBM, isotonic calibration.
- Exclusive assignment, per-entity expected-F0.5 selection, veto rules.
- A collision stress set.
- Later, a cross-encoder for uncertain pairs.
- DVC + MLflow.

My proposed changes are in `~/.claude/plans/give-me-a-complete-optimized-dijkstra.md` and are **under review**:
- **Dense retrieval as a main channel:** multilingual GPU embeddings (e.g. multilingual-e5-small, MIT) as a main blocking channel for all records, alongside DuckDB rare-token and key joins.
- **Romanisation:** turn all non-Latin text into Latin (`anyascii`, ISC licence, offline) so every string feature works across scripts.
- **Entity model:** an entity-level "has any match" model to drive the empty-vs-nonempty choice, instead of assuming pair probabilities are independent.
- **Cross-source agreement features:** true S2 and S3 matches of one S1 resemble each other.
- **Evaluation protocol:**
  - locked holdout
  - nested tuning
  - bootstrap confidence intervals
  - US↔India transfer as a France proxy
  - adversarial validation of train vs test
  - learning curves
- **Compute:** a large CPU machine (Modal / EC2 in accounts B, C or D) for features, Modal GPUs for embedding shards, Kaggle for parallel ablations.
- **Tooling:** drop DVC in favour of stage folders with done markers plus `runs.jsonl`.

---

## 8. Glossary

| Term | Meaning |
|---|---|
| Blocking / candidate generation | A cheap way to shortlist plausible S2/S3 records per S1, so we never compare all pairs |
| Recall ceiling | The share of true pairs that survive blocking. No matcher can exceed it |
| Reduction ratio | 1 − (candidates / all possible pairs). How much blocking saves |
| Singleton | An S1 with no true match |
| OOF (out-of-fold) | Predictions for each training row from a model that didn't see it. An honest internal score |
| Grouped K-fold | Folds split by S1 entity, so all of an entity's pairs stay on one side |
| Hard negatives | Non-matching candidates that look similar (top-ranked by blocking). The most useful negative examples |
| Calibration | Making a score of 0.8 mean an 80% chance of a match. Needed for expected-F0.5 decisions |
| Cross-encoder | A transformer that reads both records together and scores the pair. Accurate but slow |

---

## 9. Still unknown: checks to run on the g5

These need someone with account-A access (the g5 job queue, or JupyterLab on `test-notebook`). The output of this script fills every [est] and "unknown" above:

```bash
cd ~/SageMaker/dataset && python - <<'PY'
import polars as pl, re
r = lambda p: pl.read_csv(p, separator="\t", infer_schema_length=0, quote_char=None)
for sp in ("train", "test"):
    for i in (1, 2, 3):
        d = r(f"{sp}/{sp}_source{i}.tsv")
        nonlatin = d["business_name"].map_elements(lambda s: bool(re.search(r"[^\x00-ɏ]", s or "")), return_dtype=pl.Boolean).mean()
        print(sp, i, d.height, dict(d["country"].value_counts().iter_rows()), "empty addr", (d["business_address"].fill_null("") == "").mean(), "non-latin name", round(nonlatin, 4))
gt = r("train/train_ground_truth.tsv").with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",").list.eval(pl.element().filter(pl.element() != "")).alias("m"))
n = gt["m"].list.len()
print("singleton share", (n == 0).mean(), "matches/S1 mean", n.mean(), "p50/p90/max", n.quantile(.5), n.quantile(.9), n.max())
print("matches split S2/S3", gt["m"].explode().drop_nulls().str.slice(0, 2).value_counts())
PY
```

Also worth measuring:
- country agreement between S1 and its matches
- the address-sharing distribution per country in test
- how many matched records have names in a different script from their S1
