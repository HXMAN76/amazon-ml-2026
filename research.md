# Research: entity resolution and MLOps plan

Written 2026-09-25. Purpose: decide the modelling and infrastructure approach for the Business Entity Resolution challenge and keep that reasoning in one place. Companion to `context.md` (state of the repo and infra). Facts about this dataset come from header peeks and the problem statement; **row counts and match statistics are still pending** (stats job, see section 8).

## 1. What the task really is

- **Bipartite/multi-source linkage with a fixed reference side.** For each S1 entity, find its S2 and S3 records. S1 is already deduplicated, so this is not clustering the whole graph; it is "retrieve then verify" per S1 record.
- **Metric shapes the decision rule.** Macro F0.5 per S1 entity, singletons included (empty prediction scores 1.0 on a true singleton, any prediction scores 0.0). So: (a) a precision-first threshold, (b) an explicit "no match" outcome per entity, (c) both errors on singletons and false merges cost a whole entity's score.
- **Scale drives architecture.** About 1.2 GB of TSV per split (`train_source2` 467 MiB, `train_source3` 480 MiB), likely millions of rows per source. The number of S1 x S2/S3 pairs is far too large to enumerate, so candidate generation quality and speed decide the outcome. Literature agrees: blocking sets the recall ceiling; the matcher only trades precision against what blocking surfaced.
- **Distribution shift is built in.** `France` appears only in test. Any feature or model that leans on US/India-specific patterns (state codes, PIN formats, Indian legal forms) will degrade there. The fair-play rule forbids external gazetteers or geocoders, so generalisation has to come from string and multilingual-semantic signals.
- **Cross-script.** India S2 names can be Devanagari while S1 is Latin, addresses are Latin upper-case with noisy component order.
- **Hard limits.** MIT/Apache-2.0 licence, at most 8B parameters for the final model. No external lookups.

## 2. What the literature says

| Finding | Source | Consequence for us |
|---|---|---|
| Blocking with nearest-neighbour search over dense embeddings, sparse blocking, and string similarity joins are complementary; sparse/blocking workflows often win on time and pair completeness, dense adds recall on semantic variants | Papadakis et al., "How to reduce the search space of ER" (arXiv 2202.12521); UniBlocker (2404.14831) states dense and sparse blockers are "comparable and complementary" | Union a sparse blocker (char n-gram TF-IDF, token keys) with a dense blocker, do not choose one |
| Supervised contrastive blocking (SC-Block) builds smaller candidate sets at 99.5% pair completeness and cut a pipeline from 2.5 h to 18 min | arXiv 2303.03132 | Fine-tuning the embedder on our train labels (contrastive, with hard negatives) is worth it once a baseline exists |
| Pretrained embedding comparison over 17 datasets; embedding choice matters for blocking and matching | "Pre-trained Embeddings for ER" (2304.12329) | Benchmark 2 or 3 candidate embedders on a train sample rather than assume |
| Cross-encoders consistently beat bi-encoders for matching; generative LLM matchers do not universally beat cross-encoders, their edge is under distribution shift; larger models lean on shortcut learning | "Beyond Scale and Generation" (2607.24688), factorial study on Qwen3 family, 1,215 fine-tunes | Bi-encoder retrieve then cross-encoder rerank is the right cascade. An LLM is a tie-break for hard cases, not the main matcher |
| LLM matchers are more robust to unseen entities; fine-tuning helps small models most | Peeters and Bizer (2310.11244, 2409.08185) | For the France shift, an LLM/cross-encoder reranker on ambiguous pairs is a plausible robustness layer |
| LLM matching fails on cross-script transliteration; rule-based matchers over-match; pairwise F1 near 98-99% so pipeline parts (blocking, clustering) are the bottleneck | OpenSanctions Pairs (2603.11051) | Do explicit cross-script handling, and spend effort on blocking and decision rules rather than a bigger matcher |
| Retrieve, cheap cross-encoder auto-resolves the confident majority, escalate only ambiguous pairs; calibrate to a precision bar | "Retrieve, Match, Escalate" (2608.25037) | Same cascade shape; calibrate thresholds to precision, which fits F0.5 |
| Transitive/graph consistency: a few false positives merge many entities; cluster-editing beats plain transitive closure; multipartite matching is a multidimensional assignment problem | GraLMatch (2406.15015), TransClean (2506.04006), Multidimensional Assignment (2112.03346), cluster editing (2104.12589) | Add an assignment/consistency stage: an S2/S3 record should be claimed by one S1 entity (verify in data), and S2 to S3 agreement is extra evidence for an S1 match |
| Precision and recall need separate fixes: rule-based vetoes for precision, more diverse candidate retrieval for recall; one bad link can chain hundreds of records | "Entity Resolution in Practice" (2607.26298) | Hard vetoes (e.g. conflicting house number and PIN) as post-filter; diversify blocking keys |
| Splink (MIT, Fellegi-Sunter) links 1M records per minute on a laptop via DuckDB, scales to 100M on Spark; the runtime driver is the number of pairs after blocking; watch mega-blocks from repeated addresses | Splink docs | Cap block sizes (repeated addresses, generic names like "Christ Chapel" or "Summit Inc"); treat FS weights as an optional feature family |
| Indic transliteration exists as open models (IndicXlit, Aksharantar, open licence) and rule-based romanisation (ISO-15919); LLMs are strong but ideal transliteration is not needed for matching, only consistent phonetic mapping | Aksharantar (2205.03018), BUPS (2604.25441), romanisation study (2510.10827) | Two routes: multilingual embeddings that align scripts, and deterministic Devanagari-to-Latin romanisation as an extra sparse view |

## 3. Candidate models (licence and size checked)

| Model | Size | Licence | Use here |
|---|---|---|---|
| BAAI/bge-m3 | 568M, 1024-d, 100+ languages | MIT | Multilingual bi-encoder for blocking and features. Strong cross-lingual retrieval |
| Qwen3-Embedding-0.6B / 4B / 8B | 0.6B, 4B, 8B | Apache-2.0 | 0.6B for fast blocking over millions of records; MTEB multilingual mean 64.3 versus 59.6 for bge-m3 in its own report. 8B is at the parameter cap and too slow for 10M+ records |
| multilingual-e5-small / base | 118M / 278M | MIT | Cheaper fallback embedder, for throughput on a 4 vCPU box |
| xlm-roberta-base / mdeberta-v3-base | about 280M | MIT | Cross-encoder initialisation for supervised pair classification |
| Qwen3-Reranker 0.6B / 4B | Apache-2.0 | Pretrained reranker head; can be fine-tuned as pair scorer on our labels |
| Qwen3-8B (or similar) | 8B | Apache-2.0 | Optional LLM adjudicator on a small ambiguous slice only; at the parameter limit |
| LightGBM | n/a | MIT | Feature-based verifier and blender; already in v0 |

Note: any final model must be MIT/Apache-2.0 and up to 8B parameters, so document each pretrained model and its licence in the methodology write-up.

## 4. Recommended pipeline (target design)

```
records --> normalise (Latin accent strip only; keep Devanagari; romanised copy)
        --> BLOCKING  (union, per country group, per S1 record, top-K each)
              a) sparse: char n-gram TF-IDF on core name, name+address, on GPU or sharded
              b) keyed: rare-token / (city or PIN, house number, name prefix) keys with mega-block caps
              c) dense: Qwen3-Embedding-0.6B or bge-m3 on GPU, cosine top-K (FAISS-GPU / cuVS IVF-PQ, or exact chunked torch matmul)
        --> candidate_pairs.tsv (exactly what the matcher sees)
        --> STAGE 1 matcher: LightGBM on ~50 string/embedding/rank features (v0 already built), OOF grouped by S1
        --> STAGE 2 matcher (ambiguous band only): fine-tuned cross-encoder over "name | address" pair text
        --> DECISION: per-S1 threshold tuned for macro F0.5, singleton-aware, assignment so each S2/S3 record maps to one S1 (if data confirm), vetoes on hard conflicts
        --> matching_results.tsv
```

Why this shape:
- Cascade spends compute where it pays: cheap features resolve the easy majority, the cross-encoder handles only the ambiguous band (matches the "Retrieve, Match, Escalate" and Qwen3 factorial findings).
- The unioned blockers protect the recall ceiling. Each one is independently measurable on train (`ber.blocking.recall`).
- One-to-one and cross-source consistency exploit the task structure (S1 is deduplicated; S2 and S3 records for the same S1 should agree).
- No component calls anything external.

## 5. Accuracy levers, ordered by expected value

1. **Blocking recall at scale.** Everything downstream is bounded by it. Measure pair recall on train first; target 0.99 or better. Improve by adding keyed blocking and dense retrieval, raising K, and per-country tuning.
2. **Hard negatives.** Mine them from blocking (top-K non-matches) for both the LightGBM and the cross-encoder. The generic bottleneck in ER training data is easy negatives.
3. **Cross-script handling.** Romanised Devanagari plus multilingual embeddings; check the share of Devanagari-name records first.
4. **Threshold and decision rule for F0.5.** Tune on OOF, per source and per confidence bucket if it helps, and evaluate singleton and matched entity scores separately (`ber.metrics.breakdown`).
5. **Assignment / consistency.** Verify from ground truth how many S2/S3 records belong to more than one S1; if none, enforce exclusivity. Use S2/S3 agreement as a feature.
6. **Country generalisation.** Validate with a country hold-out: train on US only, test on India (and vice versa) as a France proxy. Prefer features that transfer (string similarity, embeddings) over country-specific ones.
7. **Fine-tuning the embedder** contrastively (SC-Block style) after the baseline gives numbers.
8. **LLM adjudication** for a tiny uncertain slice, only if the cross-encoder plateaus.

## 6. MLOps plan

Constraints: 4 vCPU and 16 GB RAM g5.xlarge notebook, A10G 24 GB, slow laptop network, 72 h window, 5 submissions per day.

- **Data flow.** Keep all bulk data in S3 and on the g5 volume. Convert TSV to Parquet once (`data/parquet/{split}/`), partitioned by source and country, so later stages read columns and shards rather than 1 GB TSVs. Laptop never touches bulk data.
- **Compute.** Notebook host (`test-notebook`) plus the S3 job queue for reproducible runs. If a long GPU job is needed, run it as a SageMaker training or processing job on `ml.g5.xlarge` (quota is for notebook usage, so check training/processing quotas first; they were 0 for several g5 sizes in this account) with managed spot and checkpoints (up to about 90% savings per AWS docs). Otherwise stay on the notebook to avoid quota round-trips.
- **Sharding and resumability.** Every heavy stage (embedding, kNN, features) works on shards keyed by (source, country, chunk) and writes to `.../work/<stage>/<shard>.parquet` with a completion marker, so a crash or interruption resumes rather than restarts. Embedding millions of records is the biggest GPU cost; cache embeddings by content hash.
- **Experiment tracking.** One append-only `runs.jsonl` per experiment (code git SHA, data version, params, blocking recall, OOF macro F0.5, singleton/matched scores, threshold, submission id and leaderboard score). MLflow (sqlite backend, MIT) is optional and can log to the same place. The guidelines require version history of all submissions.
- **Reproducibility.** Pinned `requirements.txt`, deterministic seeds, a single `python -m ber.run ...` entrypoint, config in JSON, the code zip layout required by the submission. Keep `code/business_entity_resolution` self-contained.
- **Validation discipline.** Never tune on leaderboard alone (public is a subset). Use grouped-by-S1 K-fold on train plus the country hold-out; only submit when OOF improves. Spend the 5 daily submissions on distinct hypotheses and record each.
- **Inference.** `predict` is stateless: load model files, block, score, decide, write both TSVs, validate with the official `utils/validate_submission.py`. Cache candidate features and embeddings so re-scoring after a threshold change takes seconds.
- **Cost guard.** Idle auto-stop (job-aware) already in the lifecycle config. A g5.xlarge left idle costs about $1.01/h; check `stop-notebook-instance` before leaving.
- **Model registry.** Keep model artefacts under `s3://sagemaker-us-east-1-567503593043/models/<run_id>/` with the config and OOF report next to them; no SageMaker Model Registry needed for a 3-day competition.

## 7. Risks and open questions

| Risk | Mitigation |
|---|---|
| Blocking cost at millions of rows on 16 GB / 4 vCPU | GPU chunked matmul, on-disk shards, block by country and by rare-token keys, cap block sizes |
| Dense embedding of about 15M records too slow | Start with the 0.6B model or multilingual-e5-small; embed only names (short); benchmark throughput on a sample before committing |
| France unseen | Country hold-out validation; avoid country-specific features; multilingual embeddings; calibrate thresholds so they do not depend on country |
| Mega-blocks (repeated addresses, generic names) | Cap per-key candidates; IDF weighting; treat generic names as low-evidence |
| Runner on the notebook may not be up (stats job still pending at time of writing) | Verify with a Jupyter terminal (`cat ~/onstart.log ~/jobrunner.log`); fall back to running scripts from Jupyter |
| Prohibited data use | No geocoding, gazetteers, registries, APIs. Only provided data plus public pretrained model weights (document licences) |
| CAGRA/cuVS occasionally reports zero recall on some workloads (FAISS issue 5458) | Prefer FAISS IVF or exact chunked torch matmul first; validate ANN recall against exact search on a sample |

Open questions for the data (answer via the stats job): exact row counts; fraction of S1 singletons; matches per S1 by source; whether an S2/S3 record ever belongs to more than one S1; share of Devanagari names; empty-address share; how many unmatched distractor records exist in S2/S3.

## 8. Immediate next steps

1. Read the `stats1` log (`s3://sagemaker-us-east-1-567503593043/jobs/done/stats1.log`) and fill the open questions above.
2. Convert data to Parquet on the g5 and build a train subsample that preserves match structure (all matches for a sampled set of S1 records, plus sampled distractors at the true ratio).
3. Measure blocking options on the subsample: recall@K for char n-gram TF-IDF, keyed blocking, and a dense embedder (throughput and recall).
4. Rework `ber.blocking` and features for scale (polars or Arrow, sharded, GPU top-k).
5. Train the LightGBM verifier, get the first honest OOF F0.5 and country hold-out score on the g5, then submit a valid baseline early.
6. Add the cross-encoder stage and the assignment step, re-measure, iterate.

## 9. References

- Papadakis et al., How to reduce the search space of Entity Resolution: with Blocking or Nearest Neighbor search? arXiv:2202.12521
- Pre-trained Embeddings for Entity Resolution: An Experimental Analysis. arXiv:2304.12329
- SC-Block: Supervised Contrastive Blocking within Entity Resolution Pipelines. arXiv:2303.03132
- Towards Universal Dense Blocking for Entity Resolution (UniBlocker). arXiv:2404.14831
- Beyond Scale and Generation: Understanding Language Model-based Entity Matching. arXiv:2607.24688
- Peeters and Bizer, Entity Matching using Large Language Models. arXiv:2310.11244; Fine-tuning LLMs for Entity Matching. arXiv:2409.08185
- OpenSanctions Pairs: Large-Scale Entity Matching with LLMs. arXiv:2603.11051
- Retrieve, Match, Escalate: product linking with VLM-distilled cross-encoders. arXiv:2608.25037
- GraLMatch. arXiv:2406.15015; TransClean. arXiv:2506.04006; Multidimensional Assignment Problem for multipartite ER. arXiv:2112.03346; Exploiting Transitivity Constraints for EM in KGs. arXiv:2104.12589
- Entity Resolution in Practice: Lessons from a Self-Serve Pipeline. arXiv:2607.26298
- Aksharantar / IndicXlit transliteration. arXiv:2205.03018
- Splink documentation (blocking rules, performance drivers): github.com/moj-analytical-services/splink
- Qwen3-Embedding README (MTEB comparison with BGE-M3): github.com/qwenlm/qwen3-embedding
- NVIDIA cuVS / FAISS GPU integration and the CAGRA recall issue: developer.nvidia.com blog; github.com/facebookresearch/faiss/issues/5458
- SageMaker checkpoints and managed spot training: docs.aws.amazon.com/sagemaker/latest/dg/model-checkpoints.html
- Sentence Transformers CrossEncoder training and hard-negative mining: sbert.net, huggingface.co/blog/train-reranker

## 10. Findings from the real data (2026-09-25) and what they change

Full numbers are in `context.md` section 2b. Consequences for the plan above:

1. **Exclusive ownership holds on train.** Use a global assignment step: every S2/S3 record goes to at most one S1 (highest-probability claimant). This removes a large class of false merges for free and also lets confident matches "use up" records, which helps ambiguous S1 neighbours.
2. **Address is the strongest key, name is unreliable.** A match can have a completely different name (alias or DBA) and matched records often share the exact address. But generic names (`primary care`) collide, and in France many different businesses share one address. So the matcher must combine name and address evidence and must not treat "same address" as sufficient; it must look at how many S1 entities compete for the same address (a competition feature, already partly in v0 as rank/gap and `n_prop_j`).
3. **France shift is structural, not just a new label.** Address multiplicity and tiny generic names appear only in test. Validate with a hold-out that mimics this (e.g. downweight or mask address for a US slice, or build a synthetic collision set), and keep thresholds conservative for entities that have many competitors. Precision matters double under F0.5.
4. **Multi-script Indic text** (Devanagari, Telugu, Malayalam), about 9% of India S2 names and about 9% of addresses. A multilingual embedding (bge-m3 or Qwen3-Embedding) is the practical bridge, since rule-based romanisation would need one table per script. Native-script tokens in addresses (state or city names) are also worth mapping through the embedding rather than tables.
5. **Cheap normalisation wins first:** `html.unescape`, leetspeak/digit-for-letter repair, domain-name splitting (`ipower.com` to `ipower`), DBA/`doing business as` splitting into two names, city-suffix cleanup (`CDP`). Each is a deterministic rule with no external lookup.
6. **Scale.** About 2.2M S1 vs about 10M S2+S3 in train; test 1.7M vs about 10M. With about 3.5 true matches per S1, blocking at K=20 to 30 gives 35M to 65M candidate pairs, too many for per-pair Python features on 4 vCPU. Plan: (a) train on a subsample of S1 (200k to 300k) with all their matches and the full distractor pool restricted to blocked candidates; (b) at inference, vectorised features only, stage-2 models only on an ambiguous band; (c) blocking through an inverted index over each record's rarest tokens (DuckDB or polars), not dense matmul; dense embeddings only for the non-Latin subset (about 1M records) and as an extra recall channel on names.
7. **Test-time inference is the real budget.** Decide early how many candidate pairs per S1 the inference stage can afford (target under 25 M pairs total) and design blocking to fit it.

## 11. Blocking measurements (2026-09-25)

First recall measurement of the token-index blocker on 20k train S1 against the full 10.3M-record pool (`cap_df` 800, K 30, per-type rarest-token limits n4 a4 p3 c6):

| Token types | Pair recall | S1 with all matches found | Candidates per S1 | S1 with no candidate |
|---|---|---|---|---|
| n (name words) | 0.210 | 0.112 | 13.2 | 0.528 |
| a (address words) | 0.388 | 0.240 | 21.0 | 0.250 |
| p (5-char prefixes) | 0.123 | 0.066 | 9.7 | 0.657 |
| c (name word x address word) | 0.778 | 0.486 | 15.8 | 0.002 |
| n+a | 0.559 | 0.386 | 24.9 | 0.136 |
| n+a+p | 0.557 | 0.381 | 25.2 | 0.136 |
| n+a+p+c | 0.850 (US 0.896, India 0.781) | 0.624 | 29.2 | 0.000 |

Reading:
- Single common words do not identify a business in a 10M pool. With a df cap of 800 only about 8 tokens per query survived (156k tokens for 20k queries), so half of all S1 had no usable name token. The lexical baseline needs either a higher cap or, better, more composite keys whose document frequency is naturally tiny.
- Composite keys alone beat the union of the single-word channels by a wide margin, which matches the literature view that blocking keys should combine attributes.
- Prefix tokens (`p`) add nothing on top of `n+a`, so they are candidates for removal once the new run confirms it.
- India is weaker than US (0.78 vs 0.90), consistent with non-Latin names and addresses in about 9% of India records; some of those pairs may be unreachable lexically and need a dense multilingual channel.
- Follow-up run (`blockeval2`) adds composite types `m` (name pair), `d` (address pair), `h` (house number with address word), sweeps `cap_df` at 800, 5000 and 20000, K at 30 and 100, and reports why true pairs are missing (no shared token, shared token above the cap, or lost to per-type limits and top-K). Its result decides whether the dense channel is required before gate 1.

## 12. Error analysis of baseline v0 (2026-09-25, out-of-fold on the 250k-S1 train sample)

Model: XGBoost on 42 features, exclusive assignment, threshold 0.63. Script: `code/business_entity_resolution/src/scripts/error_analysis.py`.

**Loss decomposition.** Macro F0.5 0.9377. An oracle restricted to the candidate set scores 0.9781, so blocking recall costs 0.0219 and the matcher costs 0.0404. Precision is 0.980, recall against all true pairs is 0.886, so recall is the larger loss. F0.5 versus threshold is flat around the optimum (0.9349 at 0.50, 0.9377 at 0.63, 0.9373 at 0.70): the threshold choice is not fragile.

| Segment | Macro F0.5 | Oracle on candidates |
|---|---|---|
| US | 0.959 | 0.990 |
| India | 0.905 | 0.960 |
| Singletons (5.6% of S1) | 0.920 (7.95% of them receive a wrong match) | 1.000 |
| 1 true match | 0.846 | 0.939 |
| 2 to 3 matches | 0.934 | 0.976 |
| 4 or more | 0.954 | 0.982 |
| No non-Latin match, no empty address | 0.958 | 0.992 |
| Some match has an empty address | 0.912 | 0.956 |
| Some match has a non-Latin name | 0.846 | 0.920 |
| Both | 0.785 | 0.857 |

**False positives (15,964).** 84.6% are records with no owner (look-alike distractors), 15.4% belong to another S1. Look-alikes are near-copies of an S1 with a small change: `Jarlent States LLC` versus `Jarleix States LLC`, `Olanus Fortunex` versus `Olanuz Fortunex [Ltd]` with house number 312 versus 323, `Desert Safe Virginia Inc.` versus `Desert Se Virginia Inc` with 34765 versus 34768-34772. True pairs carry the same kinds of noise (`4114` versus `4119`, a name transposition), so a single pair cannot always tell them apart; sibling evidence and rarity are the extra information.
Wrong-owner false positives are mostly non-Latin names at an identical address (`Tirupati Solutions` matched to a Telugu name owned by another S1): with no usable name similarity, the shared address dominates.

**False negatives among candidates (48,002, 5.9% of found true pairs, median p 0.36).** Typical cases: exact same name but empty address (`Wexler's Vanguard Roofing Inc`, p 0.36), non-Latin name with a Latin twin (`राम मीडिया प्राइवेट लिमिटेड` for `Ram Media Private Limited`, p 0.42), digit noise plus a street typo (`4216 45th Street` versus `4515 45TH STRETE`, p 0.05). Only 1,217 were lost to exclusive assignment.

**Blocking misses (5.9% of true pairs).** Name typo with an empty address (`Global Oneim` versus `Global Onem`), glued or handle names (`@goldenfactory`, `#l0llydigiovanni`), a completely different alias at an equal address (`Brixecto`), non-Latin name with a short address (`225, MOHALI, ਪੰਜਾਬ`), and very short addresses next to common names (`A-205, New Delhi`).

**Conclusions and ranked levers**
1. Missing features: name rarity and genericness (pool frequency of the name, IDF-weighted token coverage, exact-name flag), glued-name similarity, digit alignment (edit distance, range containment, prefix equality).
2. Cross-script: a romanised copy of non-Latin names and addresses so every string similarity applies; embeddings only if India still lags.
3. Blocking: exact core-name key, glued-name tokens, a fuzzy name channel for typo plus empty-address cases, larger K with a first-stage pruner.
4. Decision: probability calibration and per-S1 expected-F0.5 selection; the flat threshold curve says the gain is in the ranking quality, not the cut.
5. Sibling and consensus features (do other records of the same S1 agree on the digits and name variants?) address the look-alike distractors.

## 13. v1 results and the next blocking step (2026-09-25)

Added to the matcher after the section 12 analysis (63 features instead of 42): name rarity (pool count of records sharing the core name, S1 count sharing it, S1 count for the pool record's name), exact core-name flag, token coverage of names and addresses on each side, address token counts, glued-name similarity (spaces removed: partial ratio, Jaro-Winkler), digit alignment (all-digit string ratio and Levenshtein, house-number edit distance), romanised name and address similarities using `anyascii` (offline, ISC licence) and a consonant-skeleton similarity that survives vowel differences between a Latin name and the romanisation of an Indic-script name.

| Out-of-fold macro F0.5 | v0 | v1 |
|---|---|---|
| Overall | 0.9377 | **0.9551** |
| US / India | 0.959 / 0.905 | 0.968 / 0.935 |
| Singleton entities (predicted a match) | 0.920 (7.95%) | 0.959 (4.10%) |
| Entities with one true match | 0.846 | 0.883 |
| Some match has a non-Latin name | 0.846 | 0.897 |
| Some match has an empty address | 0.912 | 0.923 |
| Precision / recall vs all true pairs | 0.980 / 0.886 | 0.989 / 0.906 |
| Loss from matcher / from blocking recall | 0.0404 / 0.0219 | 0.0230 / 0.0219 |
| False positives / false negatives among candidates | 15,964 / 48,002 | 8,682 / 30,063 |

Top features by gain: `margin_p`, `rank_p` (competition between S1 entities for the same record), `house_eq`, `addr_b_empty`, `house_lev`, `num_common_frac`, `digits_ratio`, `legal_conflict`, `pin_conflict`, `name_cov_b`. The competition features dominate: knowing whether another S1 explains the record better is the strongest signal, which supports the record-centric view (each record picks one owner).

Cost note: the added Python loops (digit strings, skeletons) raised feature time from about 13 s to about 60 s per 1.5M-pair chunk; test features take about 35 minutes. Vectorising them is a pure engineering task.

Blocking is now the largest single loss (0.0219). The token types added for it, and why:
- `g` whole core name with spaces removed: handles glued and handle-style names (`@goldenfactory`, `#l0llydigiovanni`) and any exact-name pair whose tokenisation differs.
- `x` one-deletion variants of the two rarest name words (SymSpell idea: two words within one edit share a variant): handles typos when the address is empty or short (`Global Oneim` versus `Global Onem`).
- `k` consonant skeleton of name words, generated for the pool side only from romanised non-Latin names and for the S1 side from Latin names: bridges scripts at blocking time (`राम मीडिया` gives `ram midiya`, skeleton `rm md`; `Ram Media` gives `rm md`).
Job `v1c-blockeval` measures recall of these against the old token set at K 30, 40 and 60. If the gain is real, blocking is re-run for train and test, features are rebuilt and the model retrained.

Literature used for the decision layer: exact F-measure maximisation for sets of labels (Dembczynski et al., GFM: O(m^2) to O(m^3) for m candidates given the label probabilities) gives the optimal per-S1 prediction set under F-beta; calibration under domain shift (multi-domain temperature scaling, adaptive calibrator ensemble) is relevant because France is a new domain; triplet fine-tuning of embeddings on synthetic business records (arXiv 2608.16161) supports fine-tuning a bi-encoder later, if lexical features plateau.

## 14. Blocking measurement of the new token types, and the cascade (2026-09-25)

Job `v1c-blockeval` (20k train S1 against the full 10.3M pool, cap_df 800):

| Configuration | Pair recall | Candidates per S1 | Misses: unreachable / over cap / truncated |
|---|---|---|---|
| Old types (n a p c m d h), K 30 | **0.9416** | 29.9 | 0.000 / 0.018 / 0.040 |
| Old + g x k, K 30 | 0.9370 | 30.0 | 0.000 / 0.018 / 0.045 |
| Old + g x k, K 40 | 0.9410 | 39.9 | 0.000 / 0.018 / 0.040 |
| Old + g x k, K 60 | **0.9473** (US 0.970, India 0.913) | 59.6 | 0.000 / 0.018 / 0.034 |
| g x k only | 0.5055 | 22.5 | 0.000 / 0.018 / 0.476 |

Reading:
- `g` (glued whole name) and `x` (one-deletion typo variants) cover 55% and 38% of the found true pairs but are largely redundant with the old channels, and they displace better candidates in the top-K by summed IDF (K 30 gets worse). `k` (consonant skeleton) never scored: skeleton tokens of common words have document frequency far above the cap and are dropped. All three were reverted from the default pipeline.
- Lexical reachability of the true pairs is essentially complete (no shared token 0.01%; the rarest shared token has df at most 800 for 98.2%, at most 100,000 for 99.95%). The recall limit is not missing tokens; it is **ranking**: 3.4 to 4.5% of true pairs are found by tokens but cut by the per-type limits and the top-K rule, and 1.8% sit above the df cap.
- Raising K helps (K 60: +0.6 points), but doubles the pair volume for every later stage.

**Cascade blocking** (built, `stages/prune.py`): block with K 100, then re-rank with a tiny XGBoost on S1-local blocking features (score, shared tokens, per-type scores, rank, gap, ratio to the best, number of token types present) trained on the training sample's labels, and keep the best 30. The expensive string features are then computed on 30 pairs per S1 with the recall of a much larger K. The pruner's own score is dropped from the matcher's features so it cannot leak labels of the S1 it was fit on. Success criterion: pair recall at 30 kept above 0.9416, ideally approaching the K 60 and K 100 recall.
