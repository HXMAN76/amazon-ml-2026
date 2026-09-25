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
stage reads v0base/xenc0/stack0 unless explicitly pointed at them for comparison):

```
train_blockenc.py   --name blockenc0
score_blockenc.py   --name blockenc0 --split train   (then --split test)
pairs.py             --split train   (then --split test; picks up cand_dense.parquet automatically)
train_gpu.py         --name v4base --baseline (none, this IS the base model)
train_xenc2.py       --name xenc2 --baseline v4base
score_xenc.py        --name xenc2 --split train --pair-p WORK/models/v4base/oof.parquet
train_stack.py       --name stack_v4 --baseline v4base --xenc-name xenc2
```

## New piece 1: dense blocking (union, not replacement)

**Problem it targets**: token-index blocking only generates a candidate pair if source-1 and
source-2/3 share a token. It structurally cannot recover transliterated names, heavily abbreviated
names ("Intl" vs "International"), or reordered tokens with no overlap — and no downstream matcher
can rescue a pair blocking never produced. This is a *hypothesis*, not a confirmed bottleneck —
the audit below is what confirms or rules it out; nothing in this section should be built before
that audit runs.

**Design**: a small bi-encoder (`gte-small` or `bge-small`, MIT/Apache-2.0, <100M params) fine-tuned
with supervised contrastive loss on the known true pairs from `labels.parquet`. Encode every
source-1 and source-2/3 record once, build a FAISS index, retrieve top-k nearest neighbors per
source-1 record. Union these candidates with the existing token-index candidates — never subtract,
since token blocking already has known-good precision on the easy majority. Sweep k experimentally
(e.g. {5, 10, 25, 50}) against recall@k and candidate-expansion cost rather than fixing it upfront;
pick the smallest k that recovers the audit's missing true pairs without exploding candidate volume.
Keep the dense model scoped to *recall only* — it proposes candidates, it never makes the final
match/no-match call; that stays with steps 3-5.

**Model-size constraint (checked against actual corpus size)**: source1+2+3 across train+test total
~24.2M raw records that would need encoding. A decoder-based 4B-8B embedding model (e.g.
Qwen3-Embedding-4B/8B) on a single GPU realistically encodes low hundreds of records/sec, putting
full-corpus encoding at many hours — unacceptable given the "training should not take hours"
constraint. A small bi-encoder (<100M params) does thousands/sec, bringing the same job to roughly
1-2 hours; confirmed acceptable (a ~1hr encode run is within budget as long as it saturates the
GPU rather than idling on CPU-bound preprocessing — batch size should be tuned to keep GPU
utilization near 100%, same as the stage 2b cross-encoder fine-tune, not to shrink wall-clock
further). Stick to the small-bi-encoder tier only; do not substitute a multi-billion-parameter
embedding model into the blocking role regardless of its generic MTEB ranking — the cost profile
of encoding every raw record is different from the cost profile of reranking a few thousand
uncertain pairs (which is what the 305M cross-encoder in step 4 already does economically).

**Gate before this ships**: first, a zero-cost audit — join `labels.parquet` true pairs against
the current candidate set and measure what fraction are actually blocked today. Only build this if
recall is measurably short of 100%; if token blocking already recovers every true pair, this stage
adds cost (bi-encoder train + FAISS serve + larger candidate volume through steps 2-5) for zero
possible gain, and should not be built.

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
2. **Domain-knowledge tagging**: wrap house numbers and postcodes in the input text with explicit
   marker tokens (e.g. `[NUM]123[/NUM]`) before tokenization, so digit transposition/typos get a
   dedicated signal instead of being split arbitrarily by the subword tokenizer.

**Design**: same `CrossEncoderTrainer` + `BinaryCrossEntropyLoss` setup as v3's `train_xenc.py`
(already fixed and validated this session — legacy `.fit()` replaced, weights now verifiably
persist). Only the training-set construction changes: mine hard negatives via a self-join on
postcode/house-number/core-name partial matches, and preprocess `core1`/`addr` text with the
number/postcode tags before building the `datasets.Dataset`.

**Gate before this ships**: retrain `stack0`-equivalent with the v2 cross-encoder's `xs`, compare
against whichever of {v0base, v3's stack0} is currently winning.

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

## Build order (time-boxed against the 2026-09-27 deadline)

1. Cross-encoder v2 (cheap, reuses fixed code) — do this first.
2. Blocking-recall audit (near-zero cost, run regardless of time left).
3. Dense blocking — only if step 2 shows a real recall gap AND time remains after step 1.
4. Siamese-GCN — only if 1-3 all ship and time remains; otherwise not attempted.

Fallback at every step: whichever of {v0base, v3 stack0} currently holds the best gate-passing
score is what gets submitted if a v4 piece runs out of time or fails its gate.
