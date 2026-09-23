"""Sharded, resumable vision-language-model inference (Qwen-VL, SmolVLM, Idefics, InternVL, ...).

Output is written in chunks (default 500 rows) as part files, so a Kaggle session
timeout or spot interruption loses at most one chunk; rerun the same command to resume.

Example (4-bit on a 16-24GB GPU):
    python -m amlc.inference.vlm --input data/test_img.parquet --image-col image_path \
        --model Qwen/Qwen2.5-VL-7B-Instruct --load-in-4bit \
        --prompt "What is the {entity_name} of this product? Answer with value and unit only." \
        --out artifacts/vlm/test/qwen7b-v1 --shard 0 --num-shards 4

Prompt placeholders {col} are filled from the row. Check the challenge rules for
allowed model size / license before choosing a model.
"""

from __future__ import annotations

import argparse
import string
import time

import polars as pl

from amlc.data.io import read_table
from amlc.inference.shard import add_shard_args, part_path, resolve_range


def load_vlm(model_name: str, load_in_4bit: bool = False, load_in_8bit: bool = False):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    kwargs = {"device_map": "auto", "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16}
    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=kwargs["torch_dtype"],
            bnb_4bit_use_double_quant=True,
        )
    elif load_in_8bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForImageTextToText.from_pretrained(model_name, **kwargs).eval()
    processor = AutoProcessor.from_pretrained(model_name)
    if getattr(processor, "tokenizer", None) is not None:
        processor.tokenizer.padding_side = "left"  # required for batched generation
    return model, processor


def _load_image(path, max_side: int):
    from PIL import Image

    try:
        img = Image.open(path).convert("RGB")
    except Exception:  # noqa: BLE001
        return None
    img.thumbnail((max_side, max_side))  # caps visual tokens -> big speedup for Qwen-VL style models
    return img


def generate_batch(model, processor, images, prompts, max_new_tokens: int) -> list[str]:
    import torch

    texts = []
    for p in prompts:
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": p}]}]
        texts.append(processor.apply_chat_template(msgs, add_generation_prompt=True))
    inputs = processor(text=texts, images=[[im] for im in images], return_tensors="pt", padding=True)
    inputs = inputs.to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new = out[:, inputs["input_ids"].shape[1]:]
    return [s.strip() for s in processor.batch_decode(new, skip_special_tokens=True)]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--image-col", default="image_path")
    p.add_argument("--model", required=True)
    p.add_argument("--prompt", required=True, help="template, {col} placeholders filled per row")
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--chunk", type=int, default=500, help="rows per checkpointed part file")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--max-side", type=int, default=768)
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--load-in-8bit", action="store_true")
    add_shard_args(p)
    a = p.parse_args(argv)

    df = read_table(a.input).with_row_index("row")
    start, end = resolve_range(a, df.height)
    fields = [f for _, f, _, _ in string.Formatter().parse(a.prompt) if f]
    todo = [(s, min(s + a.chunk, end)) for s in range(start, end, a.chunk)
            if not part_path(a.out, s, min(s + a.chunk, end)).exists()]
    print(f"rows [{start},{end}): {len(todo)} chunks to run", flush=True)
    if not todo:
        return

    model, processor = load_vlm(a.model, a.load_in_4bit, a.load_in_8bit)
    for cs, ce in todo:
        t0 = time.time()
        rows = df.slice(cs, ce - cs).to_dicts()
        answers: list[str | None] = [None] * len(rows)
        for b in range(0, len(rows), a.batch_size):
            batch = rows[b : b + a.batch_size]
            imgs = [_load_image(r[a.image_col], a.max_side) if r[a.image_col] else None for r in batch]
            keep = [i for i, im in enumerate(imgs) if im is not None]
            if not keep:
                continue
            prompts = [a.prompt.format(**{f: batch[i][f] for f in fields}) for i in keep]
            try:
                outs = generate_batch(model, processor, [imgs[i] for i in keep], prompts, a.max_new_tokens)
            except Exception as e:  # noqa: BLE001 - e.g. OOM on one odd image: fall back to 1-by-1
                print(f"  batch failed ({e}); retrying one by one", flush=True)
                outs = []
                for i, pr in zip(keep, prompts):
                    try:
                        outs.append(generate_batch(model, processor, [imgs[i]], [pr], a.max_new_tokens)[0])
                    except Exception:  # noqa: BLE001
                        outs.append(None)
            for i, o in zip(keep, outs):
                answers[b + i] = o
        dst = part_path(a.out, cs, ce)
        dst.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({"row": [r["row"] for r in rows], "vlm_raw": answers},
                     schema={"row": pl.Int64, "vlm_raw": pl.String}).write_parquet(dst)
        print(f"  chunk [{cs},{ce}) {(ce - cs) / (time.time() - t0):.2f} rows/s", flush=True)


if __name__ == "__main__":
    main()
