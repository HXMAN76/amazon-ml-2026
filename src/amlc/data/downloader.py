"""Resumable, concurrent image downloader.

Designed for 100k-1M product image URLs (e.g. m.media-amazon.com):

* asyncio + aiohttp, bounded global and per-host concurrency
* timeout, retries with exponential backoff + jitter, honours 429/503 Retry-After
* resume: every finished URL is appended to a JSONL manifest; reruns skip done URLs
* sha256 checksum + byte size recorded per file; optional decode check
* optional on-the-fly resize (max side) to cut disk/S3 size and later decode cost
* failed URLs written to failed.csv so they can be retried separately

Usage:
    python -m amlc.data.downloader --input data/raw/train.csv --url-col image_link \
        --out data/images --concurrency 128 --max-side 512
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import aiohttp

MANIFEST = "manifest.jsonl"
FAILED = "failed.csv"
RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504}
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"


def url_to_filename(url: str) -> str:
    """Stable local filename for a URL.

    Uses the URL basename (matches the helper Amazon shipped in past challenges, so
    file names line up with any provided utils), prefixed by a short hash when the
    basename is missing or would collide across different paths.
    """
    path = urlparse(url).path
    name = os.path.basename(path)
    if not name or "." not in name:
        return hashlib.sha1(url.encode()).hexdigest()[:20] + ".jpg"
    return name


@dataclass
class Stats:
    ok: int = 0
    skipped: int = 0
    failed: int = 0
    bytes: int = 0
    started: float = field(default_factory=time.time)

    def line(self, total: int) -> str:
        done = self.ok + self.skipped + self.failed
        rate = self.ok / max(time.time() - self.started, 1e-6)
        return (
            f"{done}/{total} ok={self.ok} skip={self.skipped} fail={self.failed} "
            f"{self.bytes / 1e9:.2f}GB {rate:.1f} img/s"
        )


def load_done(out_dir: Path) -> set[str]:
    """URLs already downloaded successfully according to the manifest (and still on disk)."""
    done: set[str] = set()
    manifest = out_dir / MANIFEST
    if not manifest.exists():
        return done
    with manifest.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # torn last line after a crash
            if rec.get("status") == "ok" and (out_dir / rec["file"]).exists():
                done.add(rec["url"])
    return done


def _process_bytes(data: bytes, max_side: int | None, verify: bool) -> bytes:
    if not max_side and not verify:
        return data
    from PIL import Image

    img = Image.open(io.BytesIO(data))
    img.load()  # raises on truncated / corrupt images
    if not max_side or max(img.size) <= max_side:
        return data
    img.thumbnail((max_side, max_side))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


async def _fetch(
    session: aiohttp.ClientSession,
    url: str,
    retries: int,
    backoff: float,
) -> bytes:
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            async with session.get(url) as resp:
                if resp.status == 200:
                    return await resp.read()
                if resp.status in RETRY_STATUS and attempt < retries:
                    ra = resp.headers.get("Retry-After")
                    delay = float(ra) if ra and ra.isdigit() else backoff * 2**attempt
                    await asyncio.sleep(delay + random.uniform(0, backoff))
                    continue
                raise RuntimeError(f"HTTP {resp.status}")
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            last_err = e
            if attempt < retries:
                await asyncio.sleep(backoff * 2**attempt + random.uniform(0, backoff))
                continue
    raise RuntimeError(f"{type(last_err).__name__}: {last_err}" if last_err else "retries exhausted")


async def download_all(
    urls: list[str],
    out_dir: str | Path,
    concurrency: int = 128,
    per_host: int = 64,
    timeout: float = 20.0,
    retries: int = 4,
    backoff: float = 0.5,
    max_side: int | None = None,
    verify: bool = True,
    progress_every: int = 2000,
) -> Stats:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # dedupe while keeping order: the same image URL often appears on many rows
    unique = list(dict.fromkeys(u for u in urls if isinstance(u, str) and u.startswith("http")))
    done = load_done(out)
    todo = [u for u in unique if u not in done]
    stats = Stats(skipped=len(unique) - len(todo))
    print(f"{len(unique)} unique urls, {stats.skipped} already done, {len(todo)} to fetch", flush=True)

    # Two URLs with the same basename but different paths must not overwrite each other.
    by_name: dict[str, list[str]] = defaultdict(list)
    for u in unique:
        by_name[url_to_filename(u)].append(u)

    def fname(u: str) -> str:
        n = url_to_filename(u)
        if len(by_name[n]) > 1:
            return hashlib.sha1(u.encode()).hexdigest()[:10] + "_" + n
        return n

    manifest = (out / MANIFEST).open("a", buffering=1)
    failed = (out / FAILED).open("a", buffering=1)
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    for u in todo:
        queue.put_nowait(u)

    connector = aiohttp.TCPConnector(limit=concurrency, limit_per_host=per_host, ttl_dns_cache=300)
    client_timeout = aiohttp.ClientTimeout(total=timeout)
    loop = asyncio.get_running_loop()

    async with aiohttp.ClientSession(
        connector=connector, timeout=client_timeout, headers={"User-Agent": USER_AGENT}
    ) as session:

        async def worker() -> None:
            while True:
                try:
                    url = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                name = fname(url)
                try:
                    data = await _fetch(session, url, retries, backoff)
                    # decode/resize is CPU work: keep it off the event loop
                    data = await loop.run_in_executor(None, _process_bytes, data, max_side, verify)
                    tmp = out / (name + ".part")
                    tmp.write_bytes(data)
                    tmp.replace(out / name)  # atomic: never leave half-written images
                    rec = {
                        "url": url,
                        "file": name,
                        "status": "ok",
                        "bytes": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(),
                    }
                    manifest.write(json.dumps(rec) + "\n")
                    stats.ok += 1
                    stats.bytes += len(data)
                except Exception as e:  # noqa: BLE001 - every failure is recorded, never fatal
                    failed.write(f'"{url}","{str(e)[:200].replace(chr(34), chr(39))}"\n')
                    stats.failed += 1
                n = stats.ok + stats.failed
                if progress_every and n % progress_every == 0:
                    print(stats.line(len(unique)), flush=True)

        await asyncio.gather(*(worker() for _ in range(concurrency)))

    manifest.close()
    failed.close()
    print("DONE " + stats.line(len(unique)), flush=True)
    return stats


def read_urls(path: str, url_col: str) -> list[str]:
    import polars as pl

    df = pl.read_parquet(path) if path.endswith(".parquet") else pl.read_csv(path, infer_schema_length=0)
    return df[url_col].to_list()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="csv/parquet with an image URL column")
    p.add_argument("--url-col", default="image_link")
    p.add_argument("--out", required=True)
    p.add_argument("--concurrency", type=int, default=128)
    p.add_argument("--per-host", type=int, default=64)
    p.add_argument("--timeout", type=float, default=20.0)
    p.add_argument("--retries", type=int, default=4)
    p.add_argument("--max-side", type=int, default=None, help="resize so longest side <= N (JPEG q90)")
    p.add_argument("--no-verify", action="store_true", help="skip PIL decode check (faster)")
    p.add_argument("--start", type=int, default=0, help="row slice start (split work across machines)")
    p.add_argument("--end", type=int, default=None)
    a = p.parse_args(argv)

    urls = read_urls(a.input, a.url_col)[a.start : a.end]
    asyncio.run(
        download_all(
            urls,
            a.out,
            concurrency=a.concurrency,
            per_host=a.per_host,
            timeout=a.timeout,
            retries=a.retries,
            max_side=a.max_side,
            verify=not a.no_verify,
        )
    )


if __name__ == "__main__":
    main()
