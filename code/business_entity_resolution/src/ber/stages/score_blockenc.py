"""score_blockenc: dense retrieval with the fine-tuned bi-encoder, unioned into token blocking (v4.md stage 1).

Encodes every source-1 record and every source-2/3 record, retrieves top-K_MAX nearest pool records
per source-1 record via FAISS in one search, then reports recall@k for k in K_SWEEP against
labels.parquet (train split only) so the smallest k that materially recovers missed true pairs can
be picked without re-running the (expensive) encode+index step per k. The candidates actually
written -- in the exact schema block.py's token blocker uses (q, pid, score, ns, s_<type>...), so
pairs.py's existing `cand_*.parquet` glob picks them up with no pipeline changes -- are truncated to
--k. This stage adds candidates, it never re-implements feature computation or decides matches.
Candidates already found by token blocking are dropped (union, not duplicate).

Inputs : WORK/models/<name>/blockenc/ (fine-tuned bi-encoder weights)
         WORK/blocks/{split}/cand_*.parquet (existing token-blocking candidates, for the anti-join)
         WORK/parquet/{split}/source{1,2,3}.parquet (raw text)
         DATA/train/labels.parquet (recall@k sweep, train split only)
Outputs: WORK/blocks/{split}/cand_dense.parquet (q, pid, score, ns, s_<type>... -- s_<type> all 0,
         since these candidates carry no token-level score breakdown; score is cosine similarity)
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import polars as pl

from ber import config
from ber.stages.block import PID_BASE, TYPES
from ber.tracking import log_stage

K_SWEEP = (5, 10, 25, 50)


def main(argv: list[str] | None = None) -> None:
    """CLI: dense top-k retrieval per source-1 record, unioned with existing token-blocking candidates."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="blockenc0")
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--k", type=int, default=None, help="candidates written to disk; default is the smallest "
                     "K_SWEEP value that recovers >=95%% of what K_SWEEP's largest k recovers (train split only)")
    ap.add_argument("--batch-size", type=int, default=512)
    a = ap.parse_args(argv)
    P = config.paths()
    t0 = time.time()
    k_max = max(K_SWEEP)

    pq = P["parquet"] / a.split
    cols = ["rid", "core1", "addr"]
    s1 = pl.read_parquet(pq / "source1.parquet", columns=cols).sort("rid")
    s2 = pl.read_parquet(pq / "source2.parquet", columns=cols)
    s3 = pl.read_parquet(pq / "source3.parquet", columns=cols)
    pool = pl.concat([s2.with_columns((pl.col("rid").cast(pl.Int64) + 2 * PID_BASE).alias("pid")),
                       s3.with_columns((pl.col("rid").cast(pl.Int64) + 3 * PID_BASE).alias("pid"))])

    from sentence_transformers import SentenceTransformer

    mdl = P["work"] / "models" / a.name / "blockenc"
    model = SentenceTransformer(str(mdl))

    q_text = ("query: " + (s1["core1"] + " " + s1["addr"])).to_list()  # bge instruction prefix, query side only
    p_text = (pool["core1"] + " " + pool["addr"]).to_list()
    print(f"encoding {len(q_text)} S1 queries and {len(p_text)} pool records", flush=True)
    q_emb = model.encode(q_text, batch_size=a.batch_size, normalize_embeddings=True, show_progress_bar=False,
                          convert_to_numpy=True).astype(np.float32)
    p_emb = model.encode(p_text, batch_size=a.batch_size, normalize_embeddings=True, show_progress_bar=False,
                          convert_to_numpy=True).astype(np.float32)
    print(f"encoded in {time.time() - t0:.0f}s", flush=True)

    import faiss

    dim = p_emb.shape[1]
    nlist = max(1, min(4096, p_emb.shape[0] // 200))  # ~200 vectors/cell, capped for small pools
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(p_emb)
    index.add(p_emb)
    index.nprobe = min(32, nlist)
    scores, idx = index.search(q_emb, k_max)  # search once at k_max; sweep is a prefix slice of this, no re-search
    print(f"FAISS index built and searched (k={k_max}) in {time.time() - t0:.0f}s total", flush=True)

    chosen_k = a.k
    if a.split == "train":
        lab = pl.read_parquet(P["parquet"] / "train" / "labels.parquet").with_columns(
            (pl.col("src").cast(pl.Int64) * PID_BASE + pl.col("other_rid")).alias("pid"),
            pl.col("s1_rid").alias("q"),
        ).select("q", "pid")
        pid_matrix = pool["pid"].to_numpy()[np.where(idx >= 0, idx, 0)]  # (n_queries, k_max), 0 where unfilled
        q_of_row = s1["rid"].to_numpy()
        best_recall = 0.0
        for k in K_SWEEP:
            found = pl.DataFrame({"q": np.repeat(q_of_row, k), "pid": pid_matrix[:, :k].reshape(-1)}).unique()
            hit = lab.join(found, on=["q", "pid"], how="semi").height
            recall = hit / lab.height
            print(f"  dense recall@{k}: {hit}/{lab.height} = {recall:.4%}", flush=True)
            if k == k_max:
                best_recall = recall
            if chosen_k is None and recall >= 0.95 * best_recall and best_recall > 0:
                chosen_k = k  # smallest k reaching 95% of the k_max sweep's recall
        if chosen_k is None:
            chosen_k = k_max
        print(f"selected k={chosen_k} for candidates written to disk", flush=True)
    elif chosen_k is None:
        raise SystemExit("--k required for --split test (recall sweep only runs on train, where labels exist)")

    q_rep = np.repeat(s1["rid"].to_numpy(), chosen_k)
    pid_rep = pool["pid"].to_numpy()[idx[:, :chosen_k].reshape(-1)]
    score_rep = scores[:, :chosen_k].reshape(-1)
    valid = idx[:, :chosen_k].reshape(-1) >= 0  # faiss returns -1 for unfilled slots when pool < k
    cand = pl.DataFrame({"q": q_rep[valid], "pid": pid_rep[valid], "score": score_rep[valid].astype(np.float32)})
    cand = cand.with_columns(pl.lit(0, dtype=pl.Int16).alias("ns"))
    for t in TYPES:
        cand = cand.with_columns(pl.lit(0.0, dtype=pl.Float32).alias(f"s_{t}"))

    existing = P["work"] / "blocks" / a.split
    tok = pl.read_parquet(str(existing / "cand_[0-9]*.parquet")).select("q", "pid") if list(existing.glob("cand_[0-9]*.parquet")) else pl.DataFrame({"q": [], "pid": []}, schema={"q": pl.Int64, "pid": pl.Int64})
    before = cand.height
    cand = cand.join(tok, on=["q", "pid"], how="anti")
    print(f"dense candidates: {before} retrieved, {cand.height} new after dropping ones token blocking already has", flush=True)

    cand.write_parquet(existing / "cand_dense.parquet", compression="zstd")
    log_stage("score_blockenc", {"name": a.name, "split": a.split, "k": chosen_k},
              {"new_candidates": float(cand.height), "seconds": time.time() - t0})


if __name__ == "__main__":
    main()
