# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Nooglers  
**Team Members:** Roshan T (team leader), Hariheman V K, Sai Nivedh V, Baranidharan Selvaraj  
**Submission Date:** 25 September 2026 (draft; refreshed at the final freeze)

> Draft status (25 Sep 23:20 IST): numbers below are from the current best submitted pipeline `s4` (dense candidates plus stacked model, portal 0.953). A newer pipeline (`s5`, second dense channel) is being evaluated. Items marked *[update at freeze]* change if a later layer is adopted. The code in `code/business_entity_resolution/` reproduces every number.

---

## 1. Executive Summary

We solve entity resolution as a cascade: a weighted token-index **blocker** and two **dense retrieval channels** (a multilingual name encoder for non-Latin names, and, in the newest version, an encoder over name plus address) propose about 32 candidate S2/S3 records per Source 1 entity, an **XGBoost pair model** with 65 string, rarity, competition and script-bridging features scores every pair, and a **second-stage XGBoost** re-scores pairs using consensus evidence (how strongly other S1 entities claim the same record, how many confident records an S1 already has, whether the candidate's house number agrees with the S1's other confident records, and TF-IDF similarity). Each S2/S3 record is finally assigned to at most one S1 and a threshold tuned for F0.5 decides the matches. On a locked holdout of 150,000 training S1 entities that no model saw, macro F0.5 is **0.9708** for the current best submitted model (the first pair model alone scored 0.9565). Public leaderboard scores: 0.944 (first pair model), 0.949 (stack without dense channel), **0.953** (stack with dense channel).

---

## 2. Methodology

### 2.1 Problem Analysis
Findings from exploratory analysis of the provided data (train 2.21M S1, 5.03M S2, 5.29M S3; test 1.73M S1, 4.89M S2, 5.08M S3):
- **Match structure.** 5.6% of S1 entities have no match (singletons); the mean is 3.46 matches per S1 (up to 11: at most 5 from S2 and 6 from S3). **Every S2/S3 record belongs to at most one S1** (0 of 7.64M matched ids claimed twice), and about 27% of S2/S3 records belong to no S1 (distractors).
- **Country shift.** Training has US (60%) and India (40%); test adds France (15% of S1), for which there is no label. In France the same address occurs up to 101 times among S1 records, unlike training (at most 14).
- **Noise patterns observed in matched groups:** HTML entities, digit-for-letter substitutions (`C0mpany`), typos and transposed words, injected suffix words (`Center`, `Services`), legal-form variants (`Pvt Ltd` versus `Private Limited`), "doing business as" text and domain-style names (`ipower.com`), a completely different alias at the same address, missing address components, dropped or altered digits, `Door No` fillers, glued city suffixes (`CHICAGOCDP`), and non-Latin names and addresses (Devanagari, Telugu, Malayalam; about 9% of India names).
- **Look-alike distractors.** 84% of false positives of our first model were near-copies of an S1 that belong to no S1, typically with a house number that lost or gained a trailing digit (`6252 Golden Hook` versus `625 Golden Hook`). True pairs carry digit noise as well, so digit *relations* and agreement with the S1's other records are needed to separate them.

### 2.2 Solution Strategy
**Approach Type:** Blocking (weighted token index with a learned pruner) + gradient-boosted pair classifier + consensus stacking + exclusive assignment.  
**Core Innovation:** Treating the task as *selection among competitors*. The strongest signals are relative: how well another S1 explains the same record (`margin_pid`, `rank_p1_pid`), how many confident records an S1 already has per source, and whether a candidate's house number agrees with the S1's confident records. A second-stage model trained on out-of-fold first-stage probabilities exploits them, while the first stage stays a plain pair model.

---

## 3. Candidate Generation (Blocking)
- **Blocking keys used:** each record becomes tagged tokens, in a DuckDB inverted index over the 10.3M S2+S3 records: name words; address words and numbers; 5-character prefixes of long words; composite keys (a rare name word with a rare address word; two rare name words; two rare address words; a house number with a rare address word). A pool record's score for an S1 query is the sum of inverse document frequency over shared tokens; tokens with document frequency above 800 are ignored.
- **Cascade:** blocking keeps 100 candidates per S1 (pair recall 0.955); a small XGBoost on S1-local blocking features (score, shared tokens, per-token-type scores, rank and gap) keeps the best 30 (recall 0.947, versus 0.942 for the top 30 by score alone).
- **Candidate pairs generated:** 51,892,359 for the 1,732,544 test S1 (about 30 per S1); reduction ratio 0.99999.
- **How true matches were not lost:** on training data 99.99% of true pairs share at least one token and 98% share a token with document frequency at most 800, so the recall limit is ranking and truncation, not missing tokens. Remaining misses are empty-address pool records (33%), typo-scrambled names (24%), non-Latin names (23%) and glued or domain-style names (20%). A fine-tuned multilingual name encoder (`intfloat/multilingual-e5-small`, MIT, 118M parameters, contrastive training on true training pairs only) adds the top-5 non-Latin candidates, raising recall of non-Latin pairs from 75.9% to 85.3%. A second encoder channel over name plus address, retrieving each pool record's nearest S1 records, recovers 66% of the remaining misses for 0.5 extra candidates per S1. *[update at freeze with the final version and the candidate-set size]*

---

## 4. Matching Model

**Features used (63 in the first stage):**
- Name features: rapidfuzz ratio, partial, token-sort and token-set ratios; Jaro-Winkler; Levenshtein; token coverage on each side; exact core-name flag; glued-name similarity (spaces removed); alias and domain-label similarity; **name rarity** (how many S1 and pool records share the name); legal-form agreement and conflict.
- Address features: the same string measures; token coverage; house-number equality and edit distance; all-digit-string ratio and distance; postal-code match and conflict; empty-address flag.
- Script bridging: offline romanisation (anyascii) of non-Latin names and addresses, similarities on the romanised text, and a consonant-skeleton similarity.
- Competition: blocking score per token type, rank and gap inside the S1's list, rank and margin among the S1 entities that claim the same record.
- Second stage (consensus, on out-of-fold first-stage probabilities p1): S1-level (confident candidates per source, rank of the candidate, gap to the best) and record-level (claims on the record, margin to the best competitor), house-number relations (equal, prefix, extension, absolute difference) and digit consensus with the S1's other confident records, plus about 25 carried-over pair features. *[TF-IDF cosine and name/address consensus features under evaluation; update at freeze]*

**Model type:** XGBoost (Apache-2.0) binary classifiers on CUDA, both stages, grouped 5-fold cross-validation by S1 entity. First stage trained on 250,000 S1 entities; second stage on 400,000 other S1 entities with unbiased first-stage probabilities. No pretrained neural model in the current best pipeline.  
**Threshold selection method:** macro F0.5 maximised on out-of-fold predictions after exclusive assignment (each S2/S3 record keeps only its highest-probability S1); the threshold curve is flat around 0.65.

---

## 5. Results & Error Analysis

Locked holdout: 150,000 training S1 entities drawn once with a fixed seed from those outside every training set; 95% bootstrap intervals.

| Model | Holdout macro F0.5 | Note |
|---|---|---|
| First-stage pair model with cascade blocking (v2) | 0.9565 [0.9559, 0.9572] | public leaderboard 0.944 |
| + consensus stacking | 0.9617 | paired gain +0.0051, 95% CI [+0.0048, +0.0055] |
| + house-number relation and digit-consensus features | 0.9671 | paired gain over v2 +0.0106, 95% CI [+0.0102, +0.0110] |
| + TF-IDF cosine and name/address consensus features (`s3all`) | 0.9677 | public leaderboard 0.949 |
| + name-only dense channel for non-Latin names, France address rules (first stage 0.9598) and the same stack (`s4`) | **0.9708** | +0.0031 [+0.0028, +0.0034] over `s3all`; public leaderboard **0.953** |

The gap between the holdout and the leaderboard is what we expect from France (no training labels) and the public subset.
- **Calibration and per-S1 expected-F0.5 selection** were implemented exactly (verified against brute force) and gave no gain (+0.0001, CI includes 0): the probabilities are already calibrated (expected calibration error 0.00013).
- **Common false positives (wrong merges):** look-alike distractors, especially a house number that differs by a dropped or added trailing digit, and records with an empty address whose name equals the names of two different S1 entities.
- **Common false negatives (missed matches):** true pairs whose address is empty and whose name carries suffix or ordering noise, and non-Latin names with a short generic Latin address.
- Blocking is the largest remaining loss (about 0.02 of macro F0.5); non-Latin names account for a third of the missed pairs.

---

## 6. Conclusion
A cascade of weighted token blocking, a feature-based pair model and a consensus-stacked second stage reaches 0.967 macro F0.5 on unseen training entities. The largest lessons: relative evidence (competition between S1 entities for the same record and agreement with the S1's other confident records) matters far more than adding pairwise similarity features, and every layer must be measured on a locked holdout with a paired bootstrap because gains are small and real distributions shift (France).

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/` (all source in `src/ber`, tests in `src/tests`, helper scripts in `src/scripts`). Entry point: `make reproduce` (see the README), which runs `prepare`, `sample`, `block`, `pairs`, `train_gpu`, `predict`; the second stage is `python -m ber.stages.stack build|train|predict`. Outputs: `output/matching_results.tsv` and `output/candidate_pairs.tsv`. Parameters are in `configs/params.yaml`; runs are logged to `runs.jsonl`.

### B. Compliance and additional results
- **No external data or lookups.** Only the provided training and test files are used; the abbreviation and legal-form tables in `text.py` are hand-written string rules. Romanisation uses the offline `anyascii` package.
- **Licences and size.** XGBoost (Apache-2.0) is the final model and is far below 8 billion parameters; libraries: numpy, pandas, scipy, scikit-learn (BSD-3), polars, duckdb, rapidfuzz, pyyaml (MIT), mlflow (Apache-2.0), anyascii (ISC). *[if a pretrained multilingual encoder is adopted for candidate generation: intfloat/multilingual-e5-small, MIT, 118M parameters; update at freeze]*
- **Version history** of every run (git commit, parameters, metrics) is in `runs.jsonl` and the repository history.
- **Submission checks.** Every output file passed the organisers' `validate_submission.py` (also with `--check-ids`) and our own rule checker.
