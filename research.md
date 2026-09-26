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

## 15. Review of the teammate's v2 layered plan and the resulting priorities (2026-09-25)

The plan (layers L0 data and text views, L1 candidates, L2a pair model, L2b cross-encoder on the uncertain band, L2c stacking with consensus features, L3 owner selection per pool record with a none option, L4 per-S1 expected-F0.5 with vetoes, X1 evaluation harness with a locked holdout and bootstrap CI, X2 forensics) is sound in structure: data contracts, leakage rules, hard gates per layer, licence discipline. Corrections after comparing it with the measurements:
- Numbers were stale: out-of-fold macro F0.5 is 0.9551 (v1 features), non-Latin 0.897 versus an oracle 0.920, so the transliteration dictionary is worth at most about 0.25 points overall.
- Priorities: blocking recall (2.19 points) was missing from the plan and is the largest remaining loss; the owner layer L3 and decision layer L4 are expected to add little because competition is already in the pair model (exclusive assignment gave the identical 0.9551 score) and the threshold curve is flat. The most valuable matcher layer is L2c consensus stacking, aimed at the look-alike distractors (84% of false positives). Cross-encoder next; dictionary last.
- Design flaw to fix: L3 features (best, second, gap, competitors) computed over p1 for only the 250k sampled S1 understate competition in train (11% of S1) versus test (all S1). Fix: score every train S1 with the final model (S1 outside the sample were never trained on, so their probabilities are unbiased) and use those for record-level training.
- Contract details: use the pipeline's identifiers (`q`, `pid = src * 10_000_000 + rid`); export test `pair_p`; the locked holdout must be a separate draw of about 150k S1 because the current 250k sample is all in cross-validation; add a per-country drift monitor on test predictions as a France proxy.
- Operations: one GPU and one sequential queue mean a one-hour cross-encoder fine-tune blocks other jobs; a collaborator needs a scoped IAM user, not the root login.
Realistic remaining headroom on train-like data is about 1.5 to 2.5 points (blocking about 1 to 1.5, stacking about 0.5 to 1, decision about 0.3).

Result checks of the v1 test output: rows 1,732,544; matched S1 share France 0.948, US 0.943, India 0.929; mean matches per S1 France 3.37, US 3.33, India 3.07; share of matched ids from S2 0.489, from S3 0.511; no rule violated.

## 16. Cascade result, holdout and the first leaderboard score (2026-09-25)

- Pruner (fold 0 held out): pair recall of the best 30 is 0.9471 with the learned ranker versus 0.9415 by the blocking score; the raw K 100 ceiling is 0.9553. Out-of-fold macro F0.5 of v2 is 0.9568 (v1 0.9551); blocking loss fell from 2.19 to 1.97 points, matcher loss 2.35.
- Locked holdout (150k train S1 outside the training sample, seed 2026): v2 macro F0.5 **0.9565**, 95% bootstrap CI [0.9559, 0.9572]; precision 0.990, recall 0.909. It agrees with the out-of-fold estimate within 0.0003, so the training estimate is not inflated by leakage.
- **Leaderboard: 0.944** on the public subset for v2. The gap to the holdout, 0.0125, is the combined cost of France (15% of test S1, no training labels, crowded addresses) and the difference between the public subset and train. The test output looks healthy (France matched at 94.9% of S1 like the US at 94.3%), so the loss is more likely quality on France than a gross failure; a per-country score is not available from the portal.
- Error taxonomy of v2: false positives are 84% look-alike distractors (many with an empty address and a near-identical name that belong to another S1 or to no S1), so consensus and capacity evidence is the next lever; remaining blocking misses are non-Latin names with a short Latin address and empty-address records that lose to better-scoring candidates.

## 17. Consensus stacking and the calibration / expected-F0.5 null result (2026-09-25)

**Consensus stacking (`stages/stack.py`) ships.** Second-stage XGBoost on p1, S1-level consensus (confident candidates per source, rank inside the S1, gap to the best) and record-level consensus (claims on the same S2/S3 record, margin to the best other S1), plus 25 carried-over pair features, trained on 400k non-holdout S1 with grouped 5-fold OOF (out-of-fold macro F0.5 0.9609). Locked holdout (150k S1, paired bootstrap): v2 baseline 0.9565, stacked 0.9617, **difference +0.0051, 95% CI [+0.0048, +0.0055]**. Top features: p1, `margin_pid`, `rank_p1_pid`, `n_claim05`, `addr_b_empty`, `sum_p_pid`. The record-level competition evidence carries the gain; the S1-level capacity features add little. Test output at `runs/s1/output/`; validator and checker pass; France 94.8% of S1 matched (US 94.4%, India 93.1%).

**Calibration and per-S1 expected-F0.5 selection do not help.** Implemented exactly (`stages/expf.py`): isotonic calibration on out-of-fold stacked probabilities, then for each S1 the top-k prefix that maximises the expected F0.5, using Poisson-binomial distributions of the true positives and of the remaining true matches plus a Poisson term for matches the blocker never proposed (mu = 0.1, data estimate 0.185); verified against brute-force enumeration to 1e-6. Result on the locked holdout: 0.96180 versus 0.96169 for the threshold rule, difference +0.0001, 95% CI [-0.0002, +0.0004], not shipped. Reasons: the stacked probabilities are already calibrated (holdout expected calibration error 0.00013, 0.00009 after isotonic), the tuning grid over (mu, logit shift) is flat (0.9603 to 0.9607), and the threshold curve was flat before. There is no decision-rule slack; remaining gains must come from better ranking (features, blocking recall, model data), not from the cut.

A retrained stack (`s1b`) reproduces the holdout score of `s1` exactly (0.96169), so the estimates are stable.

## 18. Side literature check during the dense run (25 Sep evening)

Sources are arXiv abstracts only (not re-implemented). Findings relevant to our pipeline:
- Supervised contrastive blocking (SC-Block, arXiv 2303.03132) and fine-tuned embedding blockers (TriBERTa, 2411.10629) report that contrastively fine-tuned encoders beat off-the-shelf ones by a wide margin. This matches our own result: fine-tuned e5-small reached 88% union recall on non-Latin pairs at top-10.
- OpenSanctions Pairs (2603.11051): pairwise matching is near a ceiling (best LLM F1 0.99, open 14B model 0.98), rule-based matchers over-match, and LLMs "struggle with cross-script transliteration". The authors point to blocking, clustering and uncertainty-aware review as the remaining gains. Our loss decomposition (blocking about 0.02, matcher about 0.019) points the same way.
- Language-model matcher study (2607.24688, Qwen3 family): cross-encoders beat bi-encoders; for bi-encoders the embedding-oriented model variant matters most. Candidate encoder swaps: Qwen3-Embedding (Apache-2.0) or bge-m3 (MIT) instead of multilingual-e5-small.
- GraLMatch (2406.15015): a few false-positive pairs wreck group assignment; precision decides. Our exclusive ownership and consensus stacking are the same idea.
- Transliteration: AI4Bharat IndicXlit (MIT, arXiv 2205.03018) gives neural Indic-to-Latin transliteration; our `anyascii` romanisation is character-level and cruder. A better romanisation would improve the `rom_*` features and could feed romanised tokens into the lexical blocker.
- Sinkhorn or optimal-transport post-processing for one-to-one assignment (2606.02022, 2209.01847) only helps assignment when scores are not already competition-aware; ours are (margin and rank features), so we expect little.

### 18.1 Second pass (25 Sep, evening)
- **Foursquare Location Matching (Kaggle) and Shopee product matching winners** use the same shape as ours: nearest-neighbour candidate generation (top 40 or so), a first GBDT on similarity features, then a second GBDT on 200+ features including neighbour and transitive features (4th place: 20 folds). Shopee winners also post-processed predictions so their count distribution matched the training shape. Our analogue is consensus stacking; the count-shape idea led to the cardinality check below.
- **Cardinality check on `s3all` test output:** the training data has at most 5 S2 and 6 S3 matches per S1. 343 S1 exceed 5 S2 and 96 exceed 6 S3 (0.03% of S1). Capping to the top 5 S2 and top 6 S3 by probability is a free fix worth at most about +0.0002; add it in `predict.emit` at the freeze if time allows.
- **Embedding models for the Indic non-Latin channel:** BGE-M3 (MIT) leads several Indic and low-resource retrieval comparisons (Sinhala/Tamil Recall@15 about 96%, best of three on Khmer), multilingual-e5-large-instruct wins some monolingual Indic retrieval tasks, LaBSE is poor for retrieval, Jina-v3 is CC-BY-NC (not allowed). Bekko-embedding (8M active parameters, claims to beat multilingual-e5 and BGE-M3 on MMTEB retrieval) is new and unverified; licence not checked. These are sentence-retrieval benchmarks, not short business names, so fine-tuning on our own pairs matters more than the base model.
- **Learning-to-rank:** GBDT LambdaMART, YetiRank and StochasticRank are the strongest ranking objectives (2204.01500). A per-S1 `rank:pairwise` XGBoost as an extra stack member is a cheap blend candidate; our competition features already capture much of what it would add, so we expect a small gain.
- **Blocking is the intervention point:** CorpFam (2609.04269) and the "lessons from a self-serve pipeline" paper (2607.26298) both conclude that recall needs more diverse candidate retrieval and precision needs vetoes; a union of blockers beats any single one. This supports the dense union we just built.
- **F-measure decision theory:** for calibrated probabilities the optimal F1 threshold is half the best F1 (1402.1892), and decision-theoretic expected-F selection is asymptotically equivalent to plain thresholding when the model is good (1206.4625). This matches our null result for expected-F0.5 selection.


## 19. Dense channels, the miss analysis, France and the teammates' plans (25 Sep evening)

### 19.1 Results
- Name-only dense channel (multilingual-e5-small, MIT, 118M parameters; contrastive fine-tuning on 92k India S1 with 177,776 true non-Latin pairs). On the unseen holdout, recall of true non-Latin pairs rises from 75.9% (current candidates) to 81.0% (union with top-1), 83.6% (top-3), 85.3% (top-5), 88.1% (top-10). Merged at top-5: first-stage out-of-fold 0.9602 against 0.9568, holdout `v3` 0.9598 against 0.9565.
- Stacking on top (`s4`): holdout 0.9708; paired against `s3all` +0.0031 [0.0028, 0.0034]. Portal: `s4` 0.953, `s3all` 0.949, `v2` 0.944.
- Stack ablation on `v2`: TF-IDF cosine is worth about +0.0005, name/address consensus features about +0.0001 (`s3all` 0.9677, no consensus 0.9676, no TF-IDF 0.9672, `s2` 0.9671).

### 19.2 Where `s4` still loses (holdout)
Blocking loss 0.0166, matcher loss 0.0126; India 0.9569, US 0.9802. Never-proposed true pairs are 24,230 of 518,468 (4.67%): empty pool address 7,991, typo-scrambled names 5,749, non-Latin pool name 5,585, glued or domain names 4,905. Only 17.8% of the non-Latin misses lie inside the first dense channel's top-10, because their names are generic ("international") and the information sits in the address.

### 19.3 `dense_all`
Name plus address of every record, encoder fine-tuned on 250k S1 of all countries with one mined hard negative per pair, retrieval from each pool record to its nearest S1 records. `zn2` on the holdout (missed pairs recovered, extra pairs added over all train S1): top-1 no cut-off 16,025 of 24,230 for 1.09M; top-3 17,599 for 19.6M; top-5 18,356 for 39.0M; with a cosine cut-off of 0.85 only 3,167 for 47k. So each pool record's single nearest S1 carries most of the value (a record has one owner). First stage with it (`v5`, 67 features): out-of-fold 0.9757, `dall_cos` the top feature. Open risk: on the test the top-1 cosine averages 0.795 against 0.825 on train and about 40% of pool records have no owner (26% in train).

### 19.4 Country mix, difficulty and France
Test S1 mix is US 38.3%, India 46.8%, France 15.0% (train 60% and 40%). The model-based estimator (`country_expected.py`; measured bias on the holdout US +0.0068, India +0.0279) gives test F0.5 estimates France 0.9801, US 0.9824, India 0.9795, against 0.9871 and 0.9848 for US and India on the holdout: the test is harder for every country (5.8 pool records per S1 against 4.7). Adjusting the holdout for mix and difficulty gives about 0.965 for `s4`; the portal reads 0.953, an unexplained 0.012 that may be France miscalibration, public-subset noise or unmodelled misses. Output shape checks put France close to the US (matched share, mean matches, share of pool records owned); the predicted pairs have slightly weaker names in France (twice the share of very weak name pairs) and S1 that share an address with many others get fewer matches. Nothing here proves France is fine: it has no labels.

### 19.5 Candidate-set size
Organisers' email: `candidate_pairs.tsv` and its generating code count toward the final ranking, smaller candidate sets ranking higher. Our candidates average 32 per S1. Option under test (`shortlist_eval.py`, job `zo8`): keep per S1 the K best by first-stage probability inside the pipeline (the stack trains and scores only these) so that the file lists exactly what the final model scored.

### 19.6 Teammates' plans reviewed
- v4 (cross-encoder on the uncertain band; dense blocking with `bge-small-en-v1.5`): the English-only encoder cannot read Indic scripts, encoding all 24M records is unnecessary (we encode about 1.6M in two minutes), and the recall audit it proposes is already done above. The cross-encoder with hard negatives and `[NUM]` / `[POSTCODE]` tags is the only new idea; expected gain small because remaining errors are look-alike distractors and ownership conflicts. The gate must use the locked holdout (`ber.split.holdout_q`) and beat the best model here, not an older baseline.
- v5 branch (built on `sai` at `1688d18`): worth taking are decoy edit features (substitution versus indel, Hamming distance for equal lengths; 84% of false positives are look-alikes), US/Indian state-name normalisation (`Tennessee` and `TN`, `West Bengal` and its Bengali form; we saw `Oklahoma` versus `OK` in error samples) and the 5 S2 plus 6 S3 cap. Optional: exact-key channel (glued names are 20% of misses; check `dense_all` first), cross-encoder. Skip: sibling expansion (complex), per-country thresholds (our threshold curve is flat between 0.6 and 0.7). Their model names `v5` and `s5` collide with ours on a shared notebook.

### 19.7 Literature (abstract level)
Supervised contrastive blocking and fine-tuned embedding blockers beat off-the-shelf encoders (SC-Block, TriBERTa); pairwise matching is near its ceiling on multilingual benchmarks while blocking and clustering carry the remaining error (OpenSanctions Pairs, CorpFam); BGE-M3 (MIT) leads several Indic retrieval comparisons; Kaggle winners of Foursquare and Shopee used the same candidate generation, first GBDT, second GBDT with neighbour features recipe; decision-theoretic F-measure thresholds match plain thresholding for calibrated probabilities. Details in sections 18 and 18.1.


## 20. Cross-encoder, more data, and infrastructure lessons (26 Sep 2026)

### 20.1 Results (locked holdout, paired bootstrap)
- Stack tuning on the `s8` features: depth 9 with eta 0.05 (up to 1500 rounds) 0.98410 against 0.98362 (+0.00048, CI [0.00034, 0.00064]); depth 6 0.98391, depth 11 0.98385; out-of-fold agrees. The stack hit the round cap in every fold, so it was under-trained.
- Stage-two training set 1.5M S1 instead of 400k: +0.00036 [0.00019, 0.00052] alone; with decoy features, extra columns and depth 9 (`s12`) 0.98505, +0.00198 over `s6`.
- Second consensus round on the stack's probabilities (`s9`): +0.00009, CI includes 0. Extra dense neighbours for empty-address pool records (`v6`, `s7`): no gain.
- Cross-encoder (`multilingual-e5-small` with a one-logit head, fitted on 400k S1, band 0.02 to 0.98, digit tags): stack `s11` 0.98887, **+0.00525 [0.00497, 0.00555] over `s8`**; `xs` is the third most important feature. With 1.5M S1 and depth 9 (`s13`) 0.98917.
- First stage on 850k S1, depth 9, eta 0.05, up to 3000 rounds (`v7`): 0.97796 against 0.97565 (`v5`); stack on `v7` (`s15`) 0.98935, +0.00018 [0.00004, 0.00032] over `s13`.
- Reading: remaining matcher error was name noise that hand-made similarities approximate; a model that reads both records jointly captured most of it. More first-stage data and depth add little after that. The uncertain band is larger on the test (1.11 pairs per S1 for `v7`, 1.25 for `v5`) than on train (0.75 and 0.85), consistent with the test being harder.

### 20.2 What went wrong and how to avoid it
- **Stalls.** Twice (03:10 and 05:10 IST) both job lanes stopped updating their logs about 58 minutes after the notebook booted; nothing finished afterwards and status stayed InService, so the notebook was stopped and restarted (each cost 50 to 60 minutes of watching before it was noticed, first time). Cause unknown: no shell or CloudWatch data; suspects are root-volume pressure (the root disk was at 87 to 88% straight after boot), GPU contention between a scoring job and another GPU job, or something at one hour of uptime. Mitigations: per-minute diagnostic snapshot to `diag/latest.txt`, stale-log alarm in the watcher, GPU jobs serialised, `--exact-timestamps` on syncs, removal of stale live logs after a restart.
- **Silent stale code.** `aws s3 sync` skips a file whose size is unchanged, so a fix that moved a line without changing the size was never delivered to the notebook and the job failed again with the old code. Keep `--exact-timestamps`.
- **Environment split.** The `pytorch` conda env has no xgboost; steps that import `ber.stages.stack` (xenc `data`) must run in `ber`; only `train` and `score` need torch. pip installs in the `pytorch` env are lost at every notebook restart.
- **Disk.** Stack chunk folders (one per variant), the dense-merge backups and two DuckDB pool indexes (29 GB) filled the data disk to 87%; the indexes and old variants were deleted (47 GB used afterwards). Regenerating the indexes is part of a clean reproduction.


## 21. Portal result for s12, remaining loss, and the shift hypothesis (26 Sep 2026)

- **Portal:** `s12` 0.971976 (rank 402), holdout 0.98505. Gaps (holdout minus portal): v2 0.0125, s3all 0.0187, s4 0.0178, s12 0.0131. The portal gain from s4 to s12 (+0.019) exceeded the holdout gain (+0.0143), so the dense channels reduced the gap.
- **Cross-encoder 2 (`s14`):** `multilingual-e5-base` (278M) on 966k fit pairs (band 0.01 to 0.99, 7.2k confident false positives, 204k easy positives), stack `s14` 0.98986, +0.00069 over `s13`. Competition features on the refined probability (`s16`) +0.00011 over `s15`. A second cross-encoder feature `xs2` and cross-encoder scoring on `v7`'s bands are in `s17`.
- **Loss decomposition of s15** and the empty-address finding (74% of remaining misses): section 12 of `ARCHITECTURE.md`.
- **Estimated test difficulty per country (s15 probabilities, blocking misses ignored):** US 0.9940, India 0.9951, France 0.9866; uncertain best-candidate share 0.2%, 0.1%, 0.5%; mean matches per S1 3.43, 3.40, 3.60. The France row suggests over-claiming there, hence the planned France threshold experiments.
- **Shift hypothesis:** `log_cnt_s1_*`, `n_q_for_p`, rank and margin features depend on the number of S1 and pool records (test S1 count is 21% below train; pool 3% below). Remedies prepared: joint counts (`pairs.joint_counts`), density-ratio weights for the stack, adversarial validation to find the drivers.
- **Literature used:** a controlled study of matcher architectures on the Qwen3 family (arXiv 2607.24688): cross-encoders beat bi-encoders, larger models only partly narrow the gap and rely more on shortcut learning, so "bigger" is not automatically better; this motivated testing a small cross-encoder on the base model's data before scaling to `multilingual-e5-large`.
- **Infrastructure:** capacity error for `ml.g5.12xlarge` (used `ml.g5.24xlarge` for the second notebook); AWS CLI login expires after about 5 hours; stale-code and hang lessons in section 20.2.


## 22. Portal 0.981 for s17 and adversarial validation (26 Sep 2026)
- **Portal `s17` 0.981** (holdout 0.99025, gap 0.0093). Gap history 0.0178 (`s4`), 0.0131 (`s12`), 0.0093 (`s17`): each cross-encoder step helped the portal about 1.7 times as much as the holdout, because the test has more uncertain pairs per S1 (1.11 to 1.42 in the band against 0.75 to 0.96 on train).
- **Adversarial validation** (`src/scripts/adv_validation.py`, 1.5M holdout pairs against 1.5M test pairs, US and India, XGBoost on the 67 first-stage features): **AUC 0.8716**. Top features by gain: blocking scores (`s_d`, `s_h`, `score`, `s_c`, `s_m`; means nearly equal, distribution shape differs because IDF is computed inside each split's pool), `log_cnt_s1_a`, `n_cand_q` (35.8 against 39.8), `log_cnt_s1_b` (0.82 against 0.64), `n_q_for_p` (12.0 against 9.9), address token counts, `emb_cos`, `nl_name_b` (country mix). Reading: the first stage relies on set-size dependent features (competition margins, rank, counts); the test has 21% fewer S1 and 3% fewer pool records than the train.
- **Remedies in progress:** `pairs.joint_counts` (`v8`), first stage without the blocking-score and competition features (`v9`, `train_gpu --drop-cols`; stack `--drop-exact score,ns,rank_q,margin_p`), cross-encoder scoring of every shortlisted pair (`xenc data --set band_lo=0.005,band_hi=1.0`, `s22`), plus the `reemit.py` threshold variants and (backlog) per-country probe files.
- **Infrastructure:** `ml.g5.24xlarge` for the second notebook (4 GPUs, four job lanes, `bootstrap.sh` picks lane prefixes from the notebook name); notebook restart needed to add lanes; an S3-to-S3 copy failed with AccessDenied for the notebook role (tagging permission): copy through the local disk or avoid copies.
