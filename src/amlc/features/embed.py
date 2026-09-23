"""Sharded, cached image / text embedding extraction with Hugging Face models.

Cheap alternative to full VLM inference: embed once, cache to parquet, then train
GBMs / MLP heads on [image_emb | text_emb | tabular] as often as you like.

Examples:
    # image embeddings, 4 workers (run one per machine / Kaggle session)
    python -m amlc.features.embed image --input data/train_img.parquet --col image_path \
        --model google/siglip2-base-patch16-224 --out artifacts/emb/train/siglip2 --shard 0 --num-shards 4

    # text embeddings with a sentence-transformers model
    python -m amlc.features.embed text --input data/train.parquet --col catalog_content \
        --model BAAI/bge-small-en-v1.5 --out artifacts/emb/train/bge-small

    python -m amlc.inference.shard artifacts/emb/train/siglip2 --to artifacts/emb/train/siglip2.parquet
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import polars as pl

from amlc.data.io import read_table
from amlc.inference.shard import add_shard_args, part_path, resolve_range


def _device_dtype():
    import torch

    if torch.cuda.is_available():
        bf16 = torch.cuda.is_bf16_supported()
        return "cuda", torch.bfloat16 if bf16 else torch.float16
    return "cpu", torch.float32


class _ImageDS:
    def __init__(self, paths, processor):
        self.paths = paths
        self.processor = processor

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image

        p = self.paths[i]
        ok = True
        try:
            img = Image.open(p).convert("RGB")
        except Exception:  # noqa: BLE001 - missing/corrupt image -> blank + flag
            img = Image.new("RGB", (224, 224))
            ok = False
        px = self.processor(images=img, return_tensors="pt")["pixel_values"][0]
        return px, ok


def embed_images(paths: list[str | None], model_name: str, batch_size: int = 64, num_workers: int = 8):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoImageProcessor, AutoModel

    device, dtype = _device_dtype()
    model = AutoModel.from_pretrained(model_name, dtype=dtype).to(device).eval()
    processor = AutoImageProcessor.from_pretrained(model_name)
    ds = _ImageDS([p or "" for p in paths], processor)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, pin_memory=device == "cuda")

    out, oks = [], []
    t0 = time.time()
    with torch.inference_mode():
        for i, (px, ok) in enumerate(dl):
            px = px.to(device, dtype=dtype, non_blocking=True)
            if hasattr(model, "get_image_features"):  # CLIP / SigLIP family
                feats = model.get_image_features(pixel_values=px)
            else:  # ViT / DINOv2 / generic encoders
                o = model(pixel_values=px)
                feats = o.pooler_output if getattr(o, "pooler_output", None) is not None else o.last_hidden_state[:, 0]
            feats = torch.nn.functional.normalize(feats.float(), dim=-1)
            out.append(feats.cpu().numpy())
            oks.append(ok.numpy())
            if i % 50 == 0:
                print(f"  batch {i}/{len(dl)} {(i + 1) * batch_size / (time.time() - t0):.1f} img/s", flush=True)
    return np.concatenate(out), np.concatenate(oks)


def embed_texts(texts: list[str | None], model_name: str, batch_size: int = 128, max_length: int = 256):
    import torch

    device, dtype = _device_dtype()
    texts = [t or "" for t in texts]
    if "clip" in model_name.lower() or "siglip" in model_name.lower():
        # text tower of a CLIP/SigLIP model: same space as its image embeddings
        from transformers import AutoModel, AutoTokenizer

        model = AutoModel.from_pretrained(model_name, dtype=dtype).to(device).eval()
        tok = AutoTokenizer.from_pretrained(model_name)
        out = []
        with torch.inference_mode():
            for i in range(0, len(texts), batch_size):
                enc = tok(texts[i : i + batch_size], padding="max_length", truncation=True,
                          max_length=min(max_length, tok.model_max_length), return_tensors="pt").to(device)
                f = model.get_text_features(**enc)
                out.append(torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy())
        return np.concatenate(out)

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = max_length
    if device == "cuda":
        model.half()
    return model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                        show_progress_bar=True, convert_to_numpy=True).astype(np.float32)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("kind", choices=["image", "text"])
    p.add_argument("--input", required=True)
    p.add_argument("--col", required=True, help="image path column or text column")
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True, help="output dir for part files")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--max-length", type=int, default=256)
    add_shard_args(p)
    a = p.parse_args(argv)

    df = read_table(a.input)
    start, end = resolve_range(a, df.height)
    dst = part_path(a.out, start, end)
    if dst.exists():
        print(f"{dst} exists, skipping")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    col = df[a.col].slice(start, end - start).to_list()
    print(f"{a.kind} embeddings rows [{start}, {end}) with {a.model}", flush=True)

    if a.kind == "image":
        emb, ok = embed_images(col, a.model, a.batch_size, a.num_workers)
    else:
        emb = embed_texts(col, a.model, a.batch_size, a.max_length)
        ok = np.array([bool(t) for t in col])

    out = pl.DataFrame({
        "row": np.arange(start, end, dtype=np.int64),
        "ok": ok,
        "emb": emb,  # 2D array -> fixed-size list column
    })
    tmp = dst.with_suffix(".tmp")
    out.write_parquet(tmp)
    tmp.replace(dst)
    print(f"wrote {dst} {emb.shape}")


def load_matrix(path: str) -> np.ndarray:
    """Merged embedding parquet -> (n_rows, dim) float32 matrix, ordered by row."""
    df = pl.read_parquet(path).sort("row")
    return np.asarray(df["emb"].to_numpy(), dtype=np.float32)


if __name__ == "__main__":
    main()
