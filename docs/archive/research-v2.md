> **ARCHIVED (26 Sep 2026): Second research pass on top of v0; later findings are in research.md sections 18 to 22 and EXPERIMENTS.md section 9. Kept for history only; do not follow its instructions.**

# Research v2: approaches, models and priorities on top of v0

Written 2026-09-25. It extends `research.md` with a second pass focused on our actual gaps, using the numbers the v0 owner reported. Everything here is a *layer on v0*: v0's `stages/` (prepare, token-index blocking, features, pair model) stays the backbone.

## 1. Where the points are being lost

v0 figures as reported by its owner (not re-measured here):

| Metric | Value | Reading |
|---|---|---|
| Blocking recall at 30 candidates per S1 | 0.942 (US 0.970, India 0.899) | Recall caps everything downstream. India loses 10% of true pairs before any model runs |
| OOF macro F0.5 | 0.9377 | Overall |
| India / non-Latin OOF F0.5 | 0.905 / 0.846 | **Biggest gap: non-Latin names** (about 9% of India records) |
| True pairs sharing ≥ 1 token | 99.99% (96% with a document frequency ≤ 100) | Lexical blocking can reach almost all pairs; the misses are about ranking/K, not zero overlap |

So there are three levers, in order: **India/non-Latin recall and features**, **the decision layer** (precision and singletons), and **robustness for France** (unseen country, crowded addresses).

## 2. Lessons from the closest past competition: Kaggle Foursquare Location Matching (2022)

It matched POIs by name and address (plus coordinates) across noisy records at about 1M-record scale, much like our task. Consistent patterns across the top solutions:
- **Many cheap retrieval channels, unioned.**
  - The 7th-place team used 5 channels and 32 candidates: nearest neighbours, word and character name matching, embeddings. That gave a max IoU of 0.978 from retrieval alone.
  - Others used up to 128 candidates from about 7 channels (same words, same address, phone, TF-IDF, category).
- **Two-stage GBM.** A light model prunes candidates, then a heavy-feature model (LightGBM/XGBoost/CatBoost, k-fold) scores the survivors.
- **A multilingual cross-encoder as a feature.** Several top teams fine-tuned **mdeberta-v3-base** or **xlm-roberta** on pair text, Ditto-style (each field serialised with its name), and fed its score into the GBM.
- **Precision-aware training.** The 7th-place team weighted samples so false positives cost more, matching an IoU metric that punishes false merges, much like our F0.5.
- **Graph post-processing raised the ceiling.** Union-find plus pruning of weak links took the 7th-place team from 0.978 to 0.994 max IoU: matches found *through* other matches recover pairs that retrieval missed.

The last point is the biggest transferable idea for us (section 3, C).

## 3. Candidate approaches, ranked by expected gain per hour

| # | Approach | Targets | Expected gain | Cost | Risk | Verdict |
|---|---|---|---|---|---|---|
| A | **Record-level decision layer**: each S2/S3 record → owner or *none* (calibrated), an entity "has any match" model, per-S1 expected-F0.5 set choice, vetoes | precision, singletons, France crowds | Medium to high | Low: already written and tested (`decide.py`, `features.record_table/entity_table`, `model.cv_oof/crossfit_calibrate`); needs v0's candidates + OOF pair probabilities | Low | **Do first** |
| B | **Mined transliteration dictionary + romanised blocking keys**: romanise non-Latin names (`canon.romanize`, no dependencies, 9 Indic scripts); align romanised tokens to their matched S1 tokens in train pairs, then map "praivet → private", "tredars → traders"; add romanised tokens to the **blocking keys**, not just the features | India recall (0.899) and non-Latin F0.5 (0.846) | High for the 9% non-Latin slice | Low: the forensics alignment code already exists; one pass over the train pairs | Low | **Do second** |
| C | **Sibling candidate expansion** (the Foursquare graph trick): after pair scoring, for records confidently owned by S1 *s*, look up their pool neighbours (same token index, pool → pool, same country) and add *s* as a candidate for neighbours that missed it. Add a "similarity to the S1's best record" feature | recall ceiling 0.942, hard records (Devanagari, domains) | Medium to high: this raised Foursquare's ceiling by about 1.6 points | Medium: one extra blocking pass restricted to confident records (~20 min at v0's blocking speed) | Medium | **Measure first** (below), then do |
| D | **Cross-encoder on the uncertain band**: fine-tune a multilingual pair classifier on hard negatives from v0's candidates (distractors included); score only pairs the GBM is unsure about; feed the score back into the GBM | precision on ambiguous pairs; France robustness | Medium | Medium: about 30–60 min fine-tuning on the A10G, 10–20 min inference for a few million pairs | Medium | After A–C, gated on ≥ +0.005 holdout |
| E | Dense retrieval for **non-Latin records only** (multilingual-e5-base / bge-m3, or a fine-tuned bi-encoder) | India recall | Medium, but overlaps with B | Medium | Low | Only if B+C leave India behind |
| F | FP-weighted GBM training (heavier weight on negatives near singletons and crowds) | precision | Small to medium | Very low | Low | Cheap ablation during A |
| G | Indian address component parser (Shiprocket IndicBERT/TinyBERT NER, Apache-2.0) for locality/city/pincode agreement features | India precision | Small | Medium (GPU pass over ~5M India records) | Medium | Low priority |
| H | ≤ 8B LLM adjudicator (Qwen3-8B, Apache-2.0) on a tiny doubt set | edge cases, France | Small | High (slow, prompt work) | High | Skip unless everything else is done |

**How to measure C before building it (minutes, on the existing train candidates):** among true pairs (r′, s) that blocking missed, count the share where some sibling r (same owner s) *was* retrieved with s in its top 3 and r and r′ share a rare token. That share is the recall C can recover.

**Why B beats dense retrieval for India:**
- The misses are mostly spelling-bridge problems ("प्राइवेट" → "praivet" vs "private").
- A dictionary mined from the labelled pairs fixes the frequent words directly, and rule-based romanisation already handles person names (Ram, Sharma, Gupta).
- Dense retrieval over 10M records costs GPU-hours, and the blocking side already reaches 99.99% token overlap.

## 4. Model shortlist (licence and size checked against the model cards where noted)

| Use | Model | Size | Licence | Notes |
|---|---|---|---|---|
| Cross-encoder base (D) | microsoft/mdeberta-v3-base | ~280M | MIT | What Foursquare top teams used; strong multilingual |
| Cross-encoder base (D), alternative | Alibaba-NLP/gte-multilingual-reranker-base | 306M | Apache-2.0 (checked) | Already a reranker, so it may converge faster; 70+ languages; needs `trust_remote_code` |
| Cross-encoder (D), heavier | BAAI/bge-reranker-v2-m3 | 568M | Apache-2.0 | Slower; only if the base models plateau |
| Reranker/embedding (LLM-based) | Qwen3-Reranker-0.6B / Qwen3-Embedding-0.6B | 0.6B | Apache-2.0 (checked) | Strong multilingual scores; slower than encoder-only models |
| Bi-encoder (E) | intfloat/multilingual-e5-small / base; BAAI/bge-m3 | 118M / 278M / 568M | MIT | For non-Latin dense retrieval only |
| India cross-script encoder | google/muril-base-cased | ~236M | Apache-2.0 (checked) | Pretrained *with transliterated pairs*: a good base for an India-specific bi- or cross-encoder |
| Company-name embedder | dell-research-harvard/lt-wikidata-comp-multi (LinkTransformer) | ~0.3B | **Model licence not stated on the card** (package is MIT; base paraphrase-multilingual-mpnet is Apache-2.0) | Trained on Wikidata company aliases in 12 languages incl. Hindi, Bengali, French: directly on-task, but confirm the licence before use |
| Transliteration (B, optional) | AI4Bharat IndicXlit | ~11M | MIT (checked) | Best romanisation quality, but needs **fairseq** (install risk on py3.12 / torch 2.x). If used, run it once per *unique token* (a small vocabulary), not per record |
| Address NER (G) | shiprocket-ai/open-indicbert-indian-address-ner / open-tinybert-… | small | Apache-2.0 (checked) | India only |
| Phonetic codes (B/features) | jellyfish (Metaphone, NYSIIS, Match Rating) | library | MIT | Cheap extra name features for spelling variants |

**Not allowed or risky under the rules:**
- **Non-MIT/Apache licences:** Llama-based EM models (e.g. Jellyfish-8B) and Gemma-based transliterators (e.g. indic-xlit-270M) break the MIT/Apache rule.
- **libpostal** (MIT code, but OpenStreetMap-derived address knowledge) is arguably "external data augmentation". Ask the organisers through the challenge Google Form before using it.
- **TransClean-style graph cleaning** needs a labelling budget and hours of compute: too heavy for this window. Approach C captures its main benefit cheaply.

## 5. Findings that shape the design
- **NIL ("none of these") needs training examples.** Entity-linking models lose most of their NIL accuracy when trained without NIL examples; about 25% of NIL examples restores it (Zhu et al., ACL Findings 2023). Our ~25% distractors *are* those examples: the record model and any cross-encoder must train on them, not only on true-owner vs wrong-owner pairs.
- **Cross-encoders beat bi-encoders for matching.** Bigger generative models mostly help under distribution shift, which is our France case (`research.md`, §2). This supports D as a France-robustness layer rather than an LLM.
- **Transitive consistency is the main multi-source precision tool** (GraLMatch, TransClean). In our hub-and-spoke setup it reduces to: records assigned to one S1 should resemble each other. Cheap form: similarity of each candidate to the S1's current best record (C), and the average pair probability of the S1's other records (the v0 owner's stage-2 stacking).

## 6. Recommended plan (v0 stays the backbone)

| Order | Work | Owner (proposed) | Gate |
|---|---|---|---|
| 1 | v0 submission + leaderboard score | v0 owner | Calibrates OOF against leaderboard |
| 2 | A: record/entity/decision layer on v0's candidates + OOF pair probabilities | us | Holdout macro F0.5 above v0 by more than the CI (~0.003) |
| 3 | B: mined transliteration dictionary; romanised tokens in blocking keys and features | us (forensics) + v0 owner (blocking keys) | India recall up; non-Latin F0.5 up |
| 4 | C: measure the recoverable recall; if ≥ 0.5 point, build sibling expansion | us | Recall ceiling up at fixed K |
| 5 | D: cross-encoder (mdeberta-v3-base or gte-multilingual-reranker-base) on the uncertain band, as a GBM feature | whoever is free (GPU) | ≥ +0.005 holdout |
| 6 | E only if India still trails | — | — |

Evaluation throughout:
- locked holdout with a bootstrap CI
- US↔India transfer as the France proxy
- adversarial train-vs-test check
- each change an isolated ablation logged to `runs.jsonl`

## 7. Open items
- **Interface for A:** where v0 writes its candidates and OOF pair probabilities, their columns, and how its folds and holdout are defined. Needed before any code (this laptop could not `git fetch`: the SSH key for `github.com-personal` is not loaded).
- **libpostal and lt-wikidata-comp-multi:** licence and rules questions. Ask before use.

## Sources
- Foursquare 7th place write-up (Future Architect blog): https://future-architect.github.io/articles/20220720a/
- Foursquare solutions (Kaggle writeups and discussion): https://www.kaggle.com/competitions/foursquare-location-matching/writeups/re-waiwai-1st-place-solution, https://www.kaggle.com/competitions/foursquare-location-matching/discussion/336090, https://www.kaggle.com/competitions/foursquare-location-matching/writeups/bulian-ai-21st-place-solution
- 4th place code: https://github.com/TheoViel/kaggle_foursquare
- Foursquare blog, "Finding the right POI match": https://foursquare.com/resources/blog/developer/finding-the-right-poi-match/
- AnyMatch (small-LM entity matching): https://arxiv.org/html/2409.04073v1
- Fine-tuning LLMs for entity matching: https://arxiv.org/pdf/2409.08185
- OpenSanctions Pairs: https://arxiv.org/abs/2603.11051
- Learn to Not Link (NIL prediction): https://arxiv.org/html/2305.15725
- GraLMatch: https://arxiv.org/abs/2406.15015; TransClean: https://arxiv.org/html/2506.04006
- IndicXlit (MIT): https://github.com/AI4Bharat/IndicXlit
- MuRIL (Apache-2.0): https://huggingface.co/google/muril-base-cased
- gte-multilingual-reranker-base (Apache-2.0): https://huggingface.co/Alibaba-NLP/gte-multilingual-reranker-base
- Qwen3 Reranker/Embedding (Apache-2.0): https://huggingface.co/Qwen/Qwen3-Reranker-0.6B, https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
- LinkTransformer (MIT): https://github.com/dell-research-harvard/linktransformer; model: https://huggingface.co/dell-research-harvard/lt-wikidata-comp-multi
- Shiprocket Indian address NER (Apache-2.0): https://huggingface.co/shiprocket-ai/open-indicbert-indian-address-ner
- jellyfish (MIT): https://github.com/jamesturk/jellyfish
- indic-xlit-270M (Gemma-based, not allowed): https://huggingface.co/psidharth567/indic-xlit-270M
