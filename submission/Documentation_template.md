# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Nooglers  
**Team Members:** Roshan T (team leader), Hariheman V K, Sai Nivedh V, Baranidharan Selvaraj  
**Submission Date:** 27 September 2026

> Final version (freeze of 27/28 Sep). Final file: **`v8w_s29_FIN`** (stack `s29` plus the France decoding version 8), public leaderboard
> **0.987745**. Every number below is reproduced by `code/business_entity_resolution/` (`reproduce_final.sh`).

---

## 1. Executive Summary

We solve entity resolution as a cascade. A weighted token-index **blocker** with a learned pruner and two **dense retrieval channels** (multilingual
encoders over names and over name plus address) propose candidates; a first-stage **XGBoost pair model** (67 features) scores them and keeps a
short list of about **4.7 candidates per Source 1 entity**; fine-tuned multilingual **cross-encoders** read both records of every short-listed
pair; a second-stage **XGBoost stack** combines their scores with consensus evidence (how strongly other S1 entities claim the same record, how
many confident records an S1 already has, digit and house-number agreement with the S1's other records). Each S2/S3 record is assigned to at most
one S1 and an F0.5-tuned threshold decides. France, which has no training labels, gets a **structural decoding step** that removes the
country's look-alike decoys and keeps its own kinds of true copies. On a locked holdout of 150,000 training S1 entities, macro F0.5 is
**0.99088** (from 0.9565 for the first pair model); the public leaderboard moved from 0.944 to **0.987745**.

---

## 2. Methodology

### 2.1 Problem Analysis
Findings from the provided data (train 2.21M S1, 5.03M S2, 5.29M S3; test 1.73M S1, 4.89M S2, 5.08M S3):
- **Match structure.** 5.6% of S1 have no match; the mean is 3.46 matches per S1, at most 5 from S2 and 6 from S3. **Every S2/S3 record belongs
  to at most one S1**; about 27% of training S2/S3 records (41% on the test) belong to none (distractors).
- **Noise between copies:** HTML entities, digit-for-letter swaps, typos, transposed and dropped words, injected suffix words (`Center`,
  `Services`), legal-form variants, "doing business as" text and domain names, coined aliases at the same address, missing or reordered address
  components, altered house numbers, glued words, and non-Latin names (India, about 9% of names).
- **Look-alike distractors** make most false positives: near copies with a letter or house-number digit changed.
- **France (15% of test S1, no labels)** is a different generator: names are a city or brand plus a type word (`bordeaux club`), many businesses
  share one address (up to 101 S1), legal forms are often spelled out (`s a r l`), names appear as initials (`lc`) or glued (`colematernelle`), and
  its decoys are **sibling businesses whose name swaps the type word** (`nje ecole` against `nje centre`). The leaderboard confirmed that France is
  the whole gap between holdout and leaderboard (country probes: France about 0.93, US and India about 0.99).

### 2.2 Solution Strategy
**Approach Type:** blocking (token index + learned pruner + dense channels) → first-stage pair classifier → short list → cross-encoders →
consensus stack → exclusive assignment → structural decoding for the unlabelled country.  
**Core ideas:** (1) *selection among competitors*: the strongest signals are relative (another S1 explains the same record better; the S1
already has its copies in that source); (2) *read both records jointly* (cross-encoders) where hand-made similarities are ambiguous; (3) *use the
data generator's hard limits* (one owner per record, at most 5 S2 and 6 S3 matches per S1) as label-free instruments on the unlabelled country.

---

## 3. Candidate Generation (Blocking)
- **Blocking keys:** tagged tokens in a DuckDB inverted index over the 10.3M S2+S3 records: name words, address words and numbers, 5-character
  prefixes, composite keys (rare name word + rare address word, two rare name words, two rare address words, house number + rare address word).
  A pool record's score is the IDF sum over shared tokens (document-frequency cap 800); 100 candidates per S1 are kept and a small XGBoost pruner
  on S1-local blocking features keeps the best 30.
- **Dense channels:** `intfloat/multilingual-e5-small` (MIT, 118M) fine-tuned contrastively on training pairs only: (a) names only, the
  top 5 S1 for non-Latin pool names; (b) "name | address" of every record, each pool record's nearest S1 (with one mined hard negative per S1 in
  training). Channel (b) recovered 66% of the blocking misses.
- **Short list (the candidate file):** a first-stage XGBoost (67 features, trained on 850k S1) keeps per S1 the best 10 candidates with
  p1 >= 0.005: **8.2M candidate pairs, 4.74 per S1** on the test; `candidate_pairs.tsv` is exactly this list (every later model scores it).
- **Recall:** the candidate lists contain 98.6% of true holdout pairs; three quarters of the rest are pool records with an empty address whose
  name is shared by several S1 (not separable from the data).

---

## 4. Matching Model

**Features (first stage, 67):** name and address string similarities (rapidfuzz ratios, token-set/sort, Jaro-Winkler, Levenshtein), token
coverage, exact core name, glued-name similarity, alias/domain similarity, name rarity, legal-form agreement and conflict, house-number and
digit-string relations, postal codes, romanised (anyascii) and consonant-skeleton similarities for non-Latin text, blocking scores, dense
cosines and ranks, and competition (rank and margin among the S1 that claim the same record).

**Cross-encoders:** `intfloat/multilingual-e5-base` (MIT, 278M) fine-tuned on 966k training pairs from 700k S1 (uncertain band, confident false
positives as hard negatives, digit runs tagged), scoring **every short-listed pair**; a symmetric variant (both record orders averaged, two seeds)
in the final stack `s29` (as in `s27`); `Qwen/Qwen3-0.6B` (Apache-2.0, 0.6B) fine-tuned as a cross-encoder on the same training pairs and scored on
the uncertain band (second cross-encoder feature of `s29`); `multilingual-e5-base` on every short-listed pair as a further score. The S1 used to fit an encoder are excluded from later training
sets.

**Stack (second stage):** XGBoost (depth 9, eta 0.05, up to 1,500 rounds) on 1.5M S1: S1-level and record-level consensus on the first-stage and
cross-encoder probabilities, house-number and digit agreement with the S1's other confident records, TF-IDF cosines, edit-type (decoy) features,
the cross-encoder scores and about 25 carried pair features.

**Decision:** exclusive assignment (each S2/S3 record keeps its highest-probability S1), one threshold tuned for macro F0.5 on out-of-fold
predictions (0.74), at most 5 S2 and 6 S3 matches per S1.

**France decoding** (`src/scripts/france_variants.py`, test statistics only, no labels, no training on test data):
- *Type-word swap decoys* are dropped: one common word swapped into a word of the country's type vocabulary, estimated from **slot occupancy**
  (true copies must fit into an S1's free slots, decoys arrive at a rate that does not depend on how full the S1 is).
- *A protected cut-off* (0.995) for France's over-confident probabilities that spares equal names after spaced legal forms, initials, and
  France's noise-word copies (a word swapped into or added from `fils`, `groupe`, `services`, `developpement`, found from their replacement rates).
- *Protect:* an S1 the cut-off would leave empty keeps its best pair (an S1 with a true match scores 0 with an empty list).
- *Restore:* unowned short-listed pairs at the S1's own address that are one of those France copy kinds are added within the S1's free slots.
- *Final additions (version 8):* French legal-form conflicts (`sarl` against `sas`, spaced forms included) are dropped; exact names on another
  street shared by many France S1 (namesakes in the same city: France 58 per 1,000 S1 against at most 5 in the US/India) are dropped; plain coined
  aliases at the S1's exact address with p in [0.995, 0.9999) are re-added. Alias records that name the S1 are never dropped.
  Each rule was checked against the labelled holdout where the same pattern exists, by France's rate per 1,000 S1 against the US/India rate of the
  same kind, and by slot-occupancy decoy shares in France.

---

## 5. Results & Error Analysis

Locked holdout: 150,000 training S1 drawn once with a fixed seed from those outside every training set; paired bootstrap for every change.

| Model | Holdout macro F0.5 | Leaderboard |
|---|---|---|
| First-stage pair model with cascade blocking (`v2`) | 0.9565 | 0.944 |
| + consensus stacking, digit and TF-IDF features (`s3all`) | 0.9677 | 0.949 |
| + dense name channel (`s4`) | 0.9708 | 0.953 |
| + name-and-address dense channel, short list (`s5`, `s6`) | 0.9832 | |
| + decoy edit features, 1.5M S1, depth 9 (`s12`) | 0.98505 | 0.9720 |
| + e5-small cross-encoder (`s11`); e5-base cross-encoder (`s14`) | 0.98887; 0.98986 | |
| + first stage on 850k S1, both cross-encoders, refined competition (`s17`) | 0.99025 | 0.9805 |
| + e5-base cross-encoder on every short-listed pair (`s22`) | 0.99054 | |
| `s22` + France decoding (type swaps, cut-off 0.985, caps) (`s22t2c`) | 0.99054 | 0.9845 |
| symmetric cross-encoder, two seeds (`s27`) | 0.99063 | |
| `s27` + France decoding version 8 (`v8u_s27_AR`) | 0.99063 | **0.985578** |
| + Qwen3-0.6B cross-encoder score as a fourth stack feature (`s28`) | 0.99077 | |
| + Qwen3-0.6B score in the second cross-encoder slot (`s29`) + France decoding version 8 (`v8w_s29_AR`) | **0.99088** | 0.985875 |
| **Final:** `s29` + France decoding version 8 with namesake, legal-form and coined-alias lists (`v8w_s29_FIN`) | **0.99088** | **0.987745** |

- **France drives the leaderboard.** US and India score at their holdout level; France started at about 0.93. Its errors are confident
  sibling decoys (type-word swaps; about 26k pairs) and France-only true-copy forms the model scores low (initials, spelled legal forms, noise words).
- **Remaining holdout errors:** missed matches (about 0.008 of F0.5), mostly empty-address pool records whose name several S1 share; wrong merges
  (about 0.002), mostly look-alikes with a changed house number.
- **What did not help:** calibration with per-S1 expected-F0.5 selection (+0.0001), more neighbours for empty-address records, a second
  consensus round, a larger (e5-large) cross-encoder, training in a test-like universe, stack blends of seeds, an mmBERT cross-encoder (holdout
  +0.00004, interval across 0), a France pseudo-label classifier (it learned the wrong boundary outside its labelled corner).
- **Late experiment:** a French-aware e5-base cross-encoder trained on training pairs rewritten into French form (same labels; French-ized holdout
  errors 7,749 to 1,333). Used to re-add France pairs FIN had dropped, it lowered the leaderboard (0.987143): the rewrite teaches French vocabulary
  but not France's decoy semantics (in the training countries a swapped category word is noise, in France it marks a sibling business).

---

## 6. Conclusion
Relative evidence (competition for the same record, agreement with the S1's other copies) and models that read both records jointly carried the
holdout from 0.957 to 0.991. For the unseen country, the data generator's hard limits (one owner per record, at most 5 + 6 matches per S1) served
as label-free instruments to find its decoys and its own kinds of true copies; every such rule was measured before use and confirmed on the
leaderboard.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/`: all source in `src/ber` (stages `prepare`, `block`, `prune`, `dense`, `dense_all`, `pairs`, `train_gpu`,
`score_rest`, `xenc`, `stack`, `predict`), helper and decoding scripts in `src/scripts` (`france_variants.py`, `france_recall.py`,
`check_submission.py`), tests in `src/tests`. `reproduce_final.sh` runs the whole pipeline; parameters are in `configs/params.yaml`; every run is
logged to `runs.jsonl`.

### B. Compliance and additional results
- **No external data or lookups.** Only the provided files; abbreviation and legal-form tables are hand-written rules; romanisation uses the
  offline `anyascii` package. Test records were never used as training examples: the France decoding uses label-free counts on the test output.
- **Licences and size.** XGBoost (Apache-2.0); `intfloat/multilingual-e5-small` and `-base` (MIT; 118M and 278M parameters); `Qwen/Qwen3-0.6B`
  (Apache-2.0; 0.6B parameters). Total about 1.1B parameters, well below 8 billion. Libraries: numpy, pandas, scipy, scikit-learn (BSD-3), polars, duckdb, rapidfuzz, pyyaml (MIT), torch (BSD-3),
  transformers (Apache-2.0), mlflow (Apache-2.0), anyascii (ISC).
- **Version history** of every run (git commit, parameters, metrics) is in `runs.jsonl` and the repository history.
- **Submission checks.** Every output file passed the organisers' `validate_submission.py` with `--check-ids` and our rule checker.
