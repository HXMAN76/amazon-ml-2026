# v5 architecture (branch `v5/multichannel-consensus`, built on `sai` @ 1688d18)

Goal: lift the leaderboard from ~0.95 towards 0.97 (stretch 0.98) and keep it on the private board. Everything is trained on the
training data only, with open models (MIT / Apache-2.0, far below 8B parameters) and no external lookups.

## 1. Why the score stops where it does (measured)

| Fact | Number | Source |
|---|---|---|
| Test S1 mix vs train | test US 38% / **India 47% / France 15%**; train US 60% / India 40% | 15 MB samples of the test and train TSVs; `measure mix` gives the exact mix |
| India is the weak country | blocking recall India 0.899 vs US 0.970; v1 F0.5 India 0.935 vs US 0.968 | research.md §13, handoff.md §7 |
| Train-mix holdout vs leaderboard | v2 holdout 0.9565, public LB 0.944; the stack `s1` holdout 0.9617 | research.md §16, §17 |
| Ceiling set by blocking | oracle on the candidates 0.978, so 0.98 needs more recall | research.md §13 |
| Misses are ranking, not vocabulary | 0.01% of true pairs share no token; the rest are ranked out or share only frequent tokens | research.md §14 |
| Errors are look-alikes | 84% of false positives are near-copies with 1-2 changed characters / digits | research.md §12, §16 |
| No slack in the cut | calibration + exact expected-F0.5 selection: +0.0001, CI [-0.0002, +0.0004] | research.md §17 |

Re-weighting the per-country holdout scores to the test mix explains most of the gap to the leaderboard. So v5 targets India's
candidate recall, the look-alike distractors, and a decision that is judged on the test mix, not the train mix.

## 2. Pipeline (KEEP = on `sai` already, NEW / CHANGED = v5)

```
S1/S2/S3 TSV
 -> prepare     CHANGED  text.py: US / Indian state name, code or Indic-script name -> code (component-wise)
 -> candidates  union of channels, merged into WORK/blocks/{split} (channels.py, idempotent, in this order):
      token blocking K 150 -> learned pruner keeps 40      CHANGED params (was 100 -> 30)
      dense (non-Latin names), dense_all (name + address)   KEEP    fine-tuned multilingual-e5-small
      exact keys                                             NEW     stages/keys.py
 -> pass 1      pairs -> train_gpu v5a -> score_rest -> predict                                KEEP stages
 -> expand      NEW  stages/expand.py: siblings of records assigned with p >= 0.9 (rare shared key, dense_all neighbours)
 -> pass 2      pairs -> train_gpu v5b -> score_rest -> predict                                KEEP stages
 -> xenc        NEW  stages/xenc.py: cross-encoder on the band 0.05 <= p1 <= 0.95 -> xs
 -> stack s5    CHANGED stages/stack.py: consensus (KEEP) + decoy edit features + channel columns + xs
 -> decide      CHANGED decision.py / predict.emit: per-country thresholds, 5 S2 + 6 S3 cap (each only if it wins on the holdout)
 -> measure     NEW  stages/measure.py: per-country holdout, test-mix estimate, recall per channel, country transfer
```

### What each new piece does

- **State names (text.py).** `Saulsbry, Tennessee` and `SAULSBURY, TN` now share the token `tn`; `Nadia, WB`, `Nadia, West Bengal` and
  `Nadia, পশ্চিমবঙ্গ` all become `nadia wb`. It works on comma-separated components, so a lone `CT` / `FL` component is the state
  (before, the street table turned it into `court` / `floor`) while `Oak Ct` stays `oak court`. France is unchanged.
- **Exact keys (keys.py).** Pairs an S1 with pool records that have the identical cleaned name, the same glued name / alias
  (`blackstoneselect`, handles, domains) or the same name + postcode. It uses only keys shared by at most 20 (50 with a postcode) pool
  records, at most 10 per S1. Tested as token types inside the blocker, glued names displaced better candidates (§14). As a separate
  channel they only add candidates. Keys are 64-bit hashes (memory).
- **Sibling expansion (expand.py).** An S2 record and an S3 record of one business often share a Bengali or Devanagari name that is far
  from the Latin S1. Once pass 1 assigns record r to S1 q with p >= 0.9, r's near-duplicates become candidates of q, unless another
  S1 confidently owns them. The p come from out-of-fold (sample) or never-trained-on (rest) models, so train candidates are not
  optimistic.
- **Cross-encoder (xenc.py).** `Alibaba-NLP/gte-multilingual-reranker-base` (Apache-2.0, ~306M) reads both raw records together.
  - Training data: band pairs, all confident false positives (the look-alikes = hard negatives, Ditto) and sampled confident
    positives.
  - Digits are tagged `[NUM]` / `[POSTCODE]`, and Indic names carry their romanised copy.
  - Fine-tuning S1 come first from the S1 the dense encoders were fine-tuned on. Stacking already excludes those, so it loses no
    training rows.
  - It uses `CrossEncoderTrainer`; the legacy `.fit()` lost weights in v3.
- **Decoy edit features (stack.py).** After cleaning, a true variant differs mostly by dropped or added characters and words. A
  look-alike keeps the length and swaps letters (`jarlent` / `jarleix`). Features:
  - Levenshtein, the substitution estimate (Indel − Levenshtein) and indels;
  - same length, and Hamming distance when the lengths are equal;
  - length difference, first-letter match, word symmetric difference, and prefix similarity.
  These add to the existing house-number / digit-consensus features.
- **Decision (decision.py, predict.emit).**
  - Per-country thresholds (US, India) are tuned on out-of-fold data and kept only if the paired holdout CI is above 0. France has no
    training S1 and keeps the global threshold, which is the conservative choice.
  - The cap of at most 5 S2 and 6 S3 matches per S1 (the training maximum, research.md §18.1) is kept only if it does not lose.
  - Both are written into the model's `config.json`, so every output path applies them identically.
- **Measurement (measure.py).**
  - `mix`: the exact country mix of test vs train.
  - `holdout --model X`: per-country F0.5 with bootstrap CIs, and the test-mix estimate (France filled with the pooled score and,
    pessimistically, the weakest country).
  - `recall`: holdout pair recall per country, which channels found each true pair (and which found it alone), and the miss
    categories.
  - `stack train --train-countries us`: scores India with a model that never saw India, as the France proxy.

Not in v5:
- **Expected-F0.5 selection:** a measured null result (§17). It stays available as `expf`.
- **Decoder LLMs:** too slow for the time left.
- **Self-training on test rows:** the rules allow "only the provided training data", and the top teams' code is reviewed.

## 3. Run order and gates (stack `s1` / the current best stays the fallback at every step)

| Step | Command | Gate (ship only if) | Expected |
|---|---|---|---|
| 0 | `make v5_measure` on the current best (or `measure mix`, `measure holdout --model s1`) | none: confirms the test-mix estimate explains the LB | reference |
| 1 | `make v5_candidates` | `keys report` / `measure recall`: holdout recall up, India most; pairs per S1 below 1.6x v2 | +0.8 to 1.5 points of ceiling |
| 2 | `make v5_pass1 v5_expand` | `expand report`: recall gain at few added pairs | part of the above |
| 3 | `make v5_pass2` | `measure holdout --model v5b` above v2 on the test-mix estimate | |
| 4 | `make v5_xenc` | `xenc report`: average precision of xs above p1 inside the band | +0.3 to 0.5 |
| 5 | `make v5_stack` | `models/s5/holdout.json`: paired CI above 0 vs its first stage, and s5 above s1 by `scripts/paired_models.py s1 s5` | +0.4 to 0.8 with decoy features |
| 6 | submit `output/s5` | official validator `--check-ids` PASS; the final pick is the best **holdout** estimate, never the best public LB | |

Leaderboard target from the test mix: US 0.98 x 0.38 + India 0.965 x 0.47 + France 0.96 x 0.15 ≈ **0.97**.
- 0.98 needs India near today's US level.
- The India row is the one to watch in every `holdout.json`.

## 4. Private leaderboard

- **What is scored:** the private board scores the rest of the **same** test file. The README says "you submit predictions for the
  full test set in both cases". With ~1.73M S1, a random split moves the score by less than 0.001, so a drop can only be systematic.
- **Overfitting the public board:** thresholds and parameters are tuned on training S1 only, never on the leaderboard, and the final
  pick is by holdout.
- **A non-random split** (e.g. more France in private): scores are reported per country, and nothing ships that loses on a country.
- **Unseen France:** France keeps the global threshold, and the country transfer (US-only model on India) is checked.
- **Reproducibility:** fixed seeds everywhere (the key and sibling cuts are deterministically sorted), saved models, pinned
  `requirements.txt` plus `requirements-gpu.txt` (freeze on the notebook), and `make reproduce_v5` once before the final upload.

## 5. Code map

New:
- `src/ber/channels.py`
- `src/ber/stages/keys.py`, `expand.py`, `xenc.py`, `measure.py`
- `src/tests/test_v5.py`
- `requirements-gpu.txt`

Changed:
- `src/ber/text.py`: states.
- `src/ber/stages/block.py`: `INDEX_VERSION` 5.
- `src/ber/stages/stack.py`: edit features, channel columns, xs, decision layer, transfer, per-country report.
- `src/ber/decision.py`: per-country thresholds, cap.
- `src/ber/stages/predict.py`: `emit` applies them.
- `configs/params.yaml`: `prune` 150/40, `stack.base/cap/country_min_q`, `keys`, `expand`, `xenc`.
- `Makefile`: `v5_*` targets, `reproduce_v5`.
- `README.md`.
- `src/tests/test_text.py`: a Devanagari state address is now a state code.

Tests: `make test` gives 33 passed, CPU only. `test_v5_pipeline_end_to_end` runs the whole v5 pipeline on synthetic data:
- blocking is limited to 3 candidates so the new channels must recover misses;
- key merge, both passes, sibling expansion, stacking with edit features, the decision layer, the submission validator, the country
  transfer, and the measurement stage.

The synthetic data is too easy to show accuracy gains. **Every gain claimed above must still be measured on the real data (steps 0-5).**
