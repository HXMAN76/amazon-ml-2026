# v4 architecture — beyond v3

v3 ([ARCHITECTURE_v3_final.md](ARCHITECTURE_v3_final.md)) is token blocking + pointwise XGBoost +
band-gated cross-encoder stacking, trained end-to-end as `v0base` -> `xenc0` -> `stack0`. v4 is a
**separate, self-contained pipeline** that shares only the feature-computation code (`pairs.py`'s
63 features are data prep, not a model, so reusing that code isn't reusing v0base) — it trains its
own pointwise base model (`v4base`) on its own candidate set, its own cross-encoder (`xenc2`), and
its own stack (`stack_v4`), none of which read anything produced by `v0base`/`xenc0`/`stack0`. It
attacks the two things v3 deliberately deferred — blocking recall and cross-encoder training
quality (research doc [ARCHITECTURE_v3_research.md](ARCHITECTURE_v3_research.md) §2, §4).
The two pipelines are compared head-to-head, and gated the same way: ship only if paired-bootstrap
95% CI lower bound > 0 vs. whichever of {v0base, stack0} is currently the best gate-passing model.
Nothing here is assumed to win before that comparison runs — v3/stack0 stays the fallback.

## Full pipeline

```
source1/2/3 (raw records)
      |
      +---------------------------------------+
      |                                        |
[1a] token-index blocking (DuckDB, shared)  [1b] dense bi-encoder retrieval (NEW: blockenc0)
      |                                        |
      +------------------ UNION ---------------+
      |
  [2] 63 pair features (shared code, run fresh over the unioned candidate set)
      |
  [3] pointwise XGBoost, trained fresh on this candidate set  --> v4base, p1
      |
      | uncertain band, p1 in [0.15, 0.85]
      v
  [4] cross-encoder v2 (NEW: xenc2 — hard-negative mining + digit/postcode tagging,
      trained against v4base's own OOF, not v0base's)  --> xs
      |
  [5] stacking XGBoost trained fresh on v4base + xs  --> stack_v4
      |
  [6] exclusive assignment + threshold tuning (same code as v3, run on stack_v4's output)
```

Build order and commands (each stage independently runnable, `--name`/`--baseline` wired so no
stage reads v0base/xenc0/stack0 unless explicitly pointed at them for comparison). Step 1 always
runs; step 2 only runs if step 1's audit shows a material gap; steps 1-2 and step 3 (cross-encoder
v2) are independent of each other and can run in either order or in parallel:

```
1. audit_recall.py                              (zero-cost, always run first)
2. train_blockenc.py   --name blockenc0          (only if step 1 shows a material recall gap)
   score_blockenc.py   --name blockenc0 --split train   (then --split test)
3. train_xenc2.py       --name xenc2 --baseline v0base   (independent experiment, see below)
   score_xenc.py        --name xenc2 --split train --pair-p WORK/models/v0base/oof.parquet
   train_stack.py       --name stack_xenc2 --baseline v0base --xenc-name xenc2
--- only if step 2 ran and shipped: rebuild the base model on the unioned candidate set ---
   pairs.py             --split train   (then --split test; picks up cand_dense.parquet automatically)
   train_gpu.py         --name v4base
   train_xenc2.py       --name xenc2b --baseline v4base
   score_xenc.py        --name xenc2b --split train --pair-p WORK/models/v4base/oof.parquet
   train_stack.py       --name stack_v4 --baseline v4base --xenc-name xenc2b
```

## New piece 1: dense blocking (union, not replacement)

**Problem it targets**: token-index blocking only generates a candidate pair if source-1 and
source-2/3 share a token. It structurally cannot recover transliterated names, heavily abbreviated
names ("Intl" vs "International"), or reordered tokens with no overlap — and no downstream matcher
can rescue a pair blocking never produced. This is a *hypothesis*, not a confirmed bottleneck.

**Step 0, always run first, near-zero cost**: `audit_recall.py` joins `labels.parquet` true pairs
against the current token-blocking candidate set and reports (a) exact recall, (b) missed true-pair
count, and (c) a breakdown of missed cases into "likely top-K truncation" (some token overlap
exists, just not enough to rank inside block.py's per-query top-K) vs. "zero token overlap" (no
shared tokens at all — the only category a semantic/dense method could ever recover). Nothing below
this line gets built until this audit shows token blocking is **materially** below 100% recall —
"materially" meaning enough missed true pairs, concentrated enough in the zero-overlap category, to
plausibly move the final holdout F0.5 once caught. A few dozen edge-case misses do not justify the
cost below; hundreds or thousands concentrated in zero-overlap likely do.

**Design (only if the audit justifies it)**: a bi-encoder from the **<100M-parameter tier only**
(`BAAI/bge-small-en-v1.5`, MIT, ~33M params — or `gte-small`, similarly sized) fine-tuned with
supervised contrastive loss (`MultipleNegativesRankingLoss`, in-batch negatives) on the known true
pairs from `labels.parquet`. Encode every source-1 and source-2/3 record once, build a FAISS index,
retrieve top-K_MAX nearest neighbors per source-1 record in a single search, then compute recall@k
for k in {5, 10, 25, 50} as prefix slices of that one search (no repeated re-search per k) and pick
the smallest k reaching ~95% of the k=50 sweep's recall — not a fixed k=10. Union these candidates
with the existing token-index candidates — never subtract, since token blocking already has
known-good precision on the easy majority. Keep the dense model scoped to *recall only* — it
proposes candidates, it never makes the final match/no-match call; that stays with steps 3-5.
Measure, per k in the sweep: recall@k, candidate count / expansion factor, downstream
feature-computation time, downstream XGBoost training time, and final holdout macro F0.5 — the last
one is what actually decides whether any of this ships.

**Model-size constraint (checked against actual corpus size)**: source1+2+3 across train+test total
~24.2M raw records that would need encoding. A decoder-based 4B-8B embedding model (e.g.
Qwen3-Embedding-4B/8B) on a single GPU realistically encodes low hundreds of records/sec, putting
full-corpus encoding at many hours — unacceptable given the "training should not take hours"
constraint, and well outside the <100M tier this stage commits to regardless. A <100M bi-encoder
does thousands/sec, bringing the same job to roughly 1-2 hours; confirmed acceptable (a ~1hr encode
run is within budget as long as it saturates the GPU rather than idling on CPU-bound preprocessing —
batch size should be tuned to keep GPU utilization near 100%, same as the stage 2b cross-encoder
fine-tune, not to shrink wall-clock further). Do not substitute a multi-billion-parameter embedding
model into the blocking role regardless of its generic MTEB ranking — the cost profile of encoding
every raw record is different from the cost profile of reranking a few thousand uncertain pairs
(which is what the 305M cross-encoder in step 4 already does economically).

**Gate before this ships**: three conditions, all required — (1) measurable improvement on the
targeted metric (recall@k materially above token-blocking-alone recall), (2) acceptable
computational/candidate-growth cost (encode+index time within budget, downstream feature/train time
not exploding), and (3) a positive paired-bootstrap 95% CI (lower bound > 0) on final holdout macro
F0.5 versus whichever of {v0base, v3 stack0} is currently best. Failing any one of the three means
this does not ship, regardless of how the others look.

**Cost**: new training loop (bi-encoder), new serving path (FAISS), and every downstream stage
(feature computation, XGBoost train time) scales with the larger unioned candidate set. Highest
engineering cost in v4 after the audit; only justified if the audit shows a real gap.

## New piece 2: cross-encoder v2 (Ditto refinements)

**Problem it targets**: stage 2b/2c (v3) fine-tunes on a random sample of the uncertain band. Ditto
(research doc §2) reports its two largest gains come from tricks not yet applied here:

1. **Hard-negative mining**: instead of a random 87k-pair sample of the uncertain band, actively
   mine near-miss negatives — same postcode/pin but different house number, same core name but
   different city — so the cross-encoder trains on the specific confusions it needs to resolve,
   not an average of easy-and-hard uncertain pairs.
2. **Domain-knowledge tagging**: wrap digit runs in the input text with explicit marker tokens —
   `[POSTCODE]...[/POSTCODE]` for runs of 5+ digits (the same length heuristic `pairs.py` already
   uses to identify pin/postcode tokens), `[NUM]...[/NUM]` for shorter runs (house numbers, other
   digit tokens) — before tokenization, so digit transposition/typos get a dedicated signal instead
   of being split arbitrarily by the subword tokenizer.

**Design**: same `CrossEncoderTrainer` + `BinaryCrossEntropyLoss` setup as v3's `train_xenc.py`
(already fixed and validated this session — legacy `.fit()` replaced, weights now verifiably
persist). Only the training-set construction changes: mine hard negatives via a self-join on
`pin_match`/`house_eq` from the feature table (same postcode/pin but different house number, same
core name but different city), and preprocess `core1`/`addr` text with the `[NUM]`/`[POSTCODE]`
tags before building the `datasets.Dataset`. Runs as its own independent experiment against
v0base's uncertain band (`xenc2`/`stack_xenc2`) — it does not require dense blocking to have run.

**Gate before this ships**: same three conditions as dense blocking — measurable uncertain-band
accuracy improvement, acceptable retrain cost, and a positive paired-bootstrap 95% CI on final
holdout macro F0.5 versus whichever of {v0base, v3 stack0} is currently best.

**Cost**: low — reuses the fixed stage 2b/2c code paths, only the training-data construction step
is new (a polars self-join + a text-preprocessing function). Cheapest v4 addition; do this before
dense blocking if time is short.

## Explicitly not in v4

- **Siamese-GCN relational layer**: still held in reserve per v3's reasoning — the cheap version
  (`consensus_prune`) already failed the gate, and a learned GNN is materially more expensive to
  build/evaluate than the rule that lost. Only revisit if both pieces above ship and time remains
  before 2026-09-27.
- **Decoder LLM matcher (Qwen3-8B etc.)**: same reasoning as v3 — 2026 literature shows
  diminishing returns from bigger LLMs on entity matching specifically; the 305M cross-encoder
  band-gated on the uncertain 1.2% already matches the hybrid design that literature recommends.
- Both Siamese-GCN and a decoder LLM matcher stay out of v4 unless every lower-cost component
  above ships and there is substantial time remaining before 2026-09-27 — not "some" time, enough
  to build and gate something with materially higher engineering cost than anything else here.

## Build order (time-boxed against the 2026-09-27 deadline)

1. **Blocking-recall audit** (`audit_recall.py`) — near-zero cost, run first, always. Determines
   whether step 2 is worth building at all.
2. **Dense blocking** (`blockenc0` -> `score_blockenc.py`'s k-sweep) — only if step 1 shows token
   blocking is materially below 100% recall. If step 1 shows recall is already ~100%, skip this
   entirely; nothing downstream can improve on it.
3. **Cross-encoder v2** (`xenc2` — hard-negative mining + `[NUM]`/`[POSTCODE]` tagging) — independent
   of steps 1-2, runs against v0base's uncertain band regardless of what the audit finds.
4. If step 2 shipped, rebuild the base model (`v4base`) on the unioned candidate set and repeat the
   cross-encoder + stacking chain on top of it (`xenc2b` -> `stack_v4`).
5. Siamese-GCN / decoder LLM — only if 1-4 all ship (or are skipped by their own gates) and
   substantial time remains; otherwise not attempted.

Each component ships only if all three conditions hold: measurable improvement on its targeted
metric, acceptable computational/candidate-growth cost, and a positive paired-bootstrap 95% CI
(lower bound > 0) on final holdout macro F0.5 versus whichever of {v0base, v3 stack0} is currently
best. Fallback at every step: whichever of {v0base, v3 stack0} currently holds the best gate-passing
score is what gets submitted if a v4 component runs out of time or fails its gate.
