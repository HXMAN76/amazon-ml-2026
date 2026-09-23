"""Fan any sharded job out to up to 10 serverless GPUs on Modal ($30/month free per workspace).

One-time setup (each member, own Modal workspace = own $30):
    uv run modal token new
    uv run modal secret create amlc-aws AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... \
        AWS_DEFAULT_REGION=us-east-1 AMLC_BUCKET=amlc-2026-hub-567503593043 HF_TOKEN=hf_...

Presets (input/output are S3 keys inside the hub bucket):
    # download images for each shard directly from the CDN + SigLIP embeddings
    uv run modal run src/amlc/modal_app.py --preset embed-image --input-key 02-processed/test.parquet \
        --url-col image_link --model google/siglip2-base-patch16-224 --out-key 03-features/emb/test/siglip2 \
        --num-shards 10 --gpu L4

    # VLM over images, 4-bit, prompt placeholders filled from row columns
    uv run modal run src/amlc/modal_app.py --preset vlm --input-key 02-processed/test.parquet \
        --model Qwen/Qwen2.5-VL-7B-Instruct --prompt "What is the {entity_name}? Answer '<number> <unit>'." \
        --out-key 05-predictions/vlm/qwen7b-v1 --num-shards 8 --gpu L4 --extra "--load-in-4bit"

    # text embeddings
    uv run modal run src/amlc/modal_app.py --preset embed-text --input-key 00-raw/train.csv \
        --col catalog_content --model BAAI/bge-base-en-v1.5 --out-key 03-features/emb/train/bge-base --num-shards 4

    # anything else: bash with $START $END $SHARD $NUM_SHARDS $AMLC_BUCKET set
    uv run modal run src/amlc/modal_app.py --script "python -m ... --start $START --end $END" --n-rows 130000

Outputs land as part-*.parquet under s3://$AMLC_BUCKET/<out-key>/; merge with amlc.inference.shard.
Parts already in S3 are pulled first, so re-running the same command resumes after failures.
GPU $/h (Modal list price, check modal.com/pricing): T4 ~0.59, L4 ~0.80, A10G ~1.10, A100-40GB ~2.10.
"""

from __future__ import annotations

import shlex
from pathlib import Path

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0", "curl")
    .env({"HF_HOME": "/cache/hf", "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"})
)
if modal.is_local():
    # Repo paths only exist on the laptop; inside the container this module lives at /root/modal_app.py
    # and the already-built image is reused, so these steps are skipped there.
    ROOT = Path(__file__).resolve().parents[2]
    image = (
        image.pip_install_from_pyproject(str(ROOT / "pyproject.toml"), optional_dependencies=["hf", "torch"])
        .pip_install("awscli")
        .add_local_dir(ROOT / "configs", "/root/work/configs")
        .add_local_python_source("amlc")
    )

app = modal.App("amlc-2026", image=image)
hf_cache = modal.Volume.from_name("amlc-hf-cache", create_if_missing=True)
aws_secret = modal.Secret.from_name("amlc-aws")

# Background uploader so a crash/preemption keeps finished chunks; final sync after the job.
SYNC = 'aws s3 sync out/ "s3://$AMLC_BUCKET/{out}/" --only-show-errors'
PRELUDE = """set -euo pipefail
mkdir -p out
aws s3 sync "s3://$AMLC_BUCKET/{out}/" out/ --only-show-errors || true
(while true; do sleep 180; {sync} || true; done) >/dev/null 2>&1 &
SYNC_PID=$!
# the loop must die with the script, or it holds the output pipe open and the shard never returns
trap 'kill $SYNC_PID 2>/dev/null || true' EXIT
"""


@app.function(gpu="L4", timeout=8 * 3600, secrets=[aws_secret], volumes={"/cache/hf": hf_cache},
              max_containers=10, retries=modal.Retries(max_retries=2, initial_delay=10.0))
def run_shard(shard: int, script: str, num_shards: int, n_rows: int) -> str:
    import os
    import subprocess

    from amlc.inference.shard import shard_bounds

    start, end = shard_bounds(n_rows, shard, num_shards)
    env = {**os.environ, "START": str(start), "END": str(end), "SHARD": str(shard), "NUM_SHARDS": str(num_shards)}
    print(f"shard {shard}/{num_shards} rows [{start},{end})", flush=True)
    proc = subprocess.run(["bash", "-c", script], cwd="/root/work", env=env, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(proc.stdout[-4000:], flush=True)
    hf_cache.commit()
    if proc.returncode:
        raise RuntimeError(f"shard {shard} failed (exit {proc.returncode}): {proc.stdout[-800:]}")
    return f"shard {shard} ok [{start},{end})"


@app.function(secrets=[aws_secret], timeout=600)
def count_rows(input_key: str) -> int:
    import os
    import subprocess

    from amlc.data.io import read_table

    local = "/tmp/in" + Path(input_key).suffix
    subprocess.run(["aws", "s3", "cp", f"s3://{os.environ['AMLC_BUCKET']}/{input_key}", local, "--only-show-errors"],
                   check=True)
    return read_table(local).height


def build_script(preset: str, input_key: str, out_key: str, model: str, col: str, url_col: str,
                 prompt: str, extra: str) -> str:
    q = shlex.quote
    ext = Path(input_key).suffix
    fetch = f'aws s3 cp "s3://$AMLC_BUCKET/{input_key}" in{ext} --only-show-errors\n'
    head = PRELUDE.format(out=out_key, sync=SYNC.format(out=out_key)) + fetch
    images = (
        f"python -m amlc.data.downloader --input in{ext} --url-col {q(url_col)} --out img "
        "--start $START --end $END --max-side 768 --concurrency 64\n"
        f"python -m amlc.data.io --input in{ext} --images img --url-col {q(url_col)} --out in_img.parquet\n"
    )
    if preset == "embed-image":
        body = images + (f"python -m amlc.features.embed image --input in_img.parquet --col image_path "
                         f"--model {q(model)} --out out --start $START --end $END {extra}\n")
    elif preset == "embed-text":
        body = (f"python -m amlc.features.embed text --input in{ext} --col {q(col)} --model {q(model)} "
                f"--out out --start $START --end $END {extra}\n")
    elif preset == "vlm":
        body = images + (f"python -m amlc.inference.vlm --input in_img.parquet --image-col image_path "
                         f"--model {q(model)} --prompt {q(prompt)} --out out --start $START --end $END {extra}\n")
    else:
        raise SystemExit(f"unknown preset {preset!r}; use embed-image | embed-text | vlm or --script")
    return head + body + SYNC.format(out=out_key) + "\n"


@app.local_entrypoint()
def main(preset: str = "", script: str = "", input_key: str = "", out_key: str = "", model: str = "",
         col: str = "", url_col: str = "image_link", prompt: str = "", extra: str = "",
         n_rows: int = 0, num_shards: int = 10, gpu: str = "L4", shards: str = ""):
    if preset:
        if not (input_key and out_key and model):
            raise SystemExit("--preset needs --input-key, --out-key and --model")
        script = build_script(preset, input_key, out_key, model, col, url_col, prompt, extra)
    if not script:
        raise SystemExit("give --preset or --script")
    if not n_rows:
        if not input_key:
            raise SystemExit("--n-rows needed with a custom --script")
        n_rows = count_rows.remote(input_key)
    todo = [int(s) for s in shards.split(",")] if shards else list(range(num_shards))
    print(f"{n_rows} rows -> {len(todo)}/{num_shards} shards on {gpu}\n--- script ---\n{script}")

    fn = run_shard.with_options(gpu=gpu)
    failed = []
    results = list(fn.map(todo, kwargs={"script": script, "num_shards": num_shards, "n_rows": n_rows},
                          return_exceptions=True))
    for shard, res in zip(todo, results):
        if isinstance(res, Exception):
            failed.append(shard)
            print(f"FAILED shard {shard}: {res}")
        else:
            print(res)
    if failed:
        print(f"rerun failed shards with: --shards {','.join(map(str, failed))}")
