# Business Entity Resolution: blocking + gradient-boosted matcher

Solution for the Amazon ML Challenge 2026 (Business Entity Resolution). For every Source 1 (S1) business record it predicts
the matching Source 2 / Source 3 (S2/S3) records, and it also writes the candidate set the model scored. Score: macro F_0.5
over S1 entities. Everything is derived from the provided training and test TSV files only; no external data, APIs,
registries or geocoding are used, and the model uses no pretrained neural network.

## Pipeline

1. **prepare** (`stages/prepare.py`, `text.py`): reads the TSVs and writes Parquet. Text normalisation is rule based:
   HTML entity decoding, Latin-only accent stripping (Devanagari, Telugu and other scripts are preserved), lowercase,
   punctuation removal, abbreviation expansion (`corp`, `ltd`, `rd`, `st`), repair of digit-for-letter noise inside words
   (`c0mpany`), DBA/alias and domain-name splitting (`X dba Y`, `X | www.x.com`), legal forms (`pvt`, `llc`, `sarl`, `sci`)
   kept in their own field, glued city suffix cleanup, and script fractions. Also builds the label table from the ground truth.
2. **sample** (`stages/sample.py`): draws 250,000 training S1 entities as queries and assigns 5 cross-validation folds. The
   full S2+S3 pool is kept, so competition between S1 entities matches test time.
3. **block** (`stages/block.py`): candidate generation with a weighted token index in DuckDB. Every record becomes tagged
   tokens: name words, address words and numbers, 5-character prefixes of long words, and composite keys (rare name word
   with rare address word, two rare name words, two rare address words, house number with rare address word). A pool record's
   score for an S1 query is the sum of inverse document frequency over shared tokens; the top 30 per S1 are kept (about 30
   candidates per S1; recall of true pairs 0.94 on a 20k-S1 check, ceiling for the current matcher).
   Tokens with document frequency above 800 in the S2+S3 pool are ignored.
4. **pairs** (`stages/pairs.py`): 63 features per candidate pair: blocking score per token type, rank and gap within
   the S1's list and within the candidate record's list (competition between S1 entities for the same record), rapidfuzz
   name and address similarities, Jaro-Winkler, Levenshtein, alias match, house-number and postal-code agreement or conflict,
   legal-form agreement, script fractions, country agreement, name rarity (how many S1 and pool records share the name),
   exact-name flag, token coverage, glued-name similarity (spaces removed), digit alignment, and similarities on
   romanised text (offline transliteration with anyascii) plus a consonant skeleton to bridge Latin and Indic scripts.
5. **train_gpu** (`stages/train_gpu.py`): XGBoost binary classifier (CUDA when available, CPU otherwise), 5-fold cross-validation
   grouped by S1 entity, then the decision rule is tuned on the out-of-fold predictions.
6. **predict** (`stages/predict.py`, `decision.py`): scores all test candidates, keeps for every S2/S3 record only its
   highest-probability S1 (records belong to at most one S1 in the training data), applies the tuned threshold, and writes
   `matching_results.tsv` and `candidate_pairs.tsv` (the candidates are exactly the pairs the model scored), then validates.

Every S1 entity, including entities of a country never seen in training (France), gets exactly one row; empty list means no match.

## Reproduce

Requirements: Python 3.12, about 60 GB of scratch disk, 15 GB RAM or more; a CUDA GPU makes training take under 3 minutes
(CPU works, slower). Measured on an AWS g5.xlarge (4 vCPU, 15 GB RAM, NVIDIA A10G):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export BER_DATA=/path/to/dataset     # folder containing train/ and test/ (the TSV files from the challenge)
export BER_WORK=/path/to/work        # scratch and outputs
make reproduce                       # runs every stage below; outputs in $BER_WORK/output/v1/
```

`make reproduce` runs, in order (timings measured on the g5):

| Step | Command | Time |
|---|---|---|
| prepare + sample | `make sample` | about 9 min (24M records) |
| block test | `python -m ber.stages.block --split test` | about 18 min (includes building the pool index) |
| block train | `python -m ber.stages.block --split train --all-train` | about 13 min |
| features train | `python -m ber.stages.pairs --split train` | about 6 min |
| train | `python -m ber.stages.train_gpu --name v1` | about 3 min on GPU |
| features test | `python -m ber.stages.pairs --split test` | about 35 min |
| predict | `python -m ber.stages.predict --name v1` | about 2 min |

Outputs: `$BER_WORK/output/v1/matching_results.tsv` (submitted to the leaderboard) and
`$BER_WORK/output/v1/candidate_pairs.tsv`. Validate with the organisers' script and with the bundled checker, which tests every
rule of the statement (format, one row per S1, ids exist, matches are a subset of candidates, one owner per record) and prints
per-country statistics:

```bash
python3 utils/validate_submission.py --matching $BER_WORK/output/v1/matching_results.tsv \
    --candidate $BER_WORK/output/v1/candidate_pairs.tsv --test-dir $BER_DATA/test --check-ids
python src/scripts/check_submission.py $BER_WORK/output/v1 $BER_DATA/test
```

`make` reruns a stage only when its parameters or source changed. Parameters for every stage are in `configs/params.yaml`.
Run tracking (parameters, metrics, timings) goes to `$BER_WORK/runs/runs.jsonl` and an MLflow sqlite database.
Tests (`make test`, 16 tests) include end-to-end runs on a small synthetic dataset, including the output checker.

## Layout

All Python source is under `src/`; `configs/`, `Makefile`, `README.md` and `requirements.txt` are at the package root.

```
src/ber/                 package
  text.py                text normalisation (entities, leetspeak, aliases, domains, legal forms, romanisation)
  config.py stamp.py tracking.py   parameters, stage cache stamps, run logging
  decision.py            exclusive assignment, threshold tuning, vectorised macro F_0.5
  validate.py            local re-implementation of the format rules (raw-line parsing)
  synth.py               small synthetic dataset used by the tests
  stages/                prepare, sample, block, block_eval, pairs, train_gpu, predict
src/scripts/             qa_prepare.py (raw vs normalised text), error_analysis.py (loss decomposition and error
                         taxonomy of a trained model), check_submission.py (rule checker for the output files)
src/tests/               unit and end-to-end tests (`make test`)
configs/params.yaml      all tunables (`block.types` selects the token types used for the submitted run)
Makefile                 stage DAG and `make reproduce`
```

## Results (training data, cross-validated)

Out-of-fold macro F_0.5 on 250,000 training S1 entities: 0.9551 at threshold 0.65 (v0 with 42 features: 0.9377); recall ceiling of blocking 0.94. Test
run: 51,892,359 candidate pairs for 1,732,544 S1 entities, 93.9% of S1 receive at least one match, mean 3.24 matches per S1
(training truth: 94.4% and 3.46). Candidate recall by country on training: US 0.970, India 0.899. France has no training
data, so its behaviour is unvalidated.

## Licences and constraints

No pretrained model is used. Libraries and their licences: numpy (BSD-3), pandas (BSD-3), polars (MIT), duckdb (MIT),
rapidfuzz (MIT), anyascii (ISC), xgboost (Apache-2.0), pyyaml (MIT), mlflow (Apache-2.0), pytest (MIT). The final model is an XGBoost
gradient-boosted tree ensemble (Apache-2.0, far below 8B parameters). The abbreviation and legal-form tables in `text.py`
are hand-written string rules, not external data lookups.
