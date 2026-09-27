#!/usr/bin/env python3
"""GPU-side entry point of a SageMaker training job that runs BER pipeline stages (started by aws/sm/sm.py).

Inside the job container (AWS PyTorch DLC, Python 3.12, CUDA matched to the host):
  1. print the machine (GPU, CPUs, RAM, disk) and install requirements.txt into the image's Python;
  2. restore WORK from S3 (BER_WORK_S3), so every job continues where the previous one stopped;
  3. run each stage in BER_STAGES with `make <stage>` (or `smoke`: GPU check + the test suite), printing timestamped markers;
  4. after every stage, save WORK to S3 (mirror, deletions included; rebuildable large files excluded), so a failure keeps finished work;
  5. a heartbeat thread prints GPU, RAM and disk use every BER_HEARTBEAT seconds: the live log shows the job is alive and how busy.
A failing stage writes /opt/ml/output/failure, which SageMaker shows as the job's FailureReason.

Environment (set by sm.py): BER_STAGES, BER_WORK_S3, BER_CODE (package root), BER_GIT_SHA, BER_HEARTBEAT (default 120).
Local test hooks: BER_ML_ROOT (instead of /opt/ml), BER_MAKE_CMD (instead of `make`), BER_SKIP_INSTALL=1.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

ML = Path(os.environ.get("BER_ML_ROOT", "/opt/ml"))
CODE = Path(os.environ.get("BER_CODE", "/opt/code/business_entity_resolution"))
DATA = Path(os.environ.get("BER_DATA", ML / "input" / "data" / "dataset"))  # the notebook queue sets BER_DATA
# large files that the pipeline rebuilds on demand: never shipped to S3 (DuckDB pool index ~10 GB per split, raw K-150 lists)
SYNC_EXCLUDE = ["*.duckdb", "*.duckdb.wal", "duckdb_tmp/*", "blocks/train_raw/*", "blocks/test_raw/*", "*.tmp", ".ckpt_stage/*"]
# what a stage checkpoint keeps (files the stage wrote): model weights, configs, reports, predictions of the decision layer,
# measurements and the submission file. Rebuildable bulk (candidate shards, feature parts, p1 of the rest of train) stays out.
CKPT_INCLUDE = ["models/*", "dense/model_ft/*", "dense_all/model_ft/*", "xenc/model/*", "xenc/train_q.npy", "measure/*.json",
                "runs/runs.jsonl", "output/*/matching_results.tsv", "keys/*/pairs.parquet", "expand/*/pairs.parquet",
                "dense_all_train_q.npy", "dense_all2/model_ft/*", "dense_all2_train_q.npy", "output/*/candidate_pairs.tsv"]
CKPT_EXCLUDE = ["models/*/p1_rest/*", "models/*/ckpt/*", "xenc/ckpt/*"]


def log(msg: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def run(cmd: str, cwd: Path | None = None, env: dict | None = None) -> int:
    log(f"$ {cmd}")
    return subprocess.run(cmd, shell=True, cwd=cwd, env=env).returncode


def pick_work_dir() -> Path:
    """BER_WORK when set (notebook queue); else the biggest local filesystem the job offers (ml.g5: the NVMe under /opt/ml)."""
    if os.environ.get("BER_WORK"):
        w = Path(os.environ["BER_WORK"])
        w.mkdir(parents=True, exist_ok=True)
        return w
    cands = [ML / "checkpoints", ML / "model", ML / "output" / "data", Path("/tmp")]
    best = max((p for p in cands if p.exists() or p.parent.exists()), key=lambda p: shutil.disk_usage(p if p.exists() else p.parent).free)
    w = best / "ber_work"
    w.mkdir(parents=True, exist_ok=True)
    return w


def machine_summary(work: Path) -> None:
    log(f"python {sys.version.split()[0]} | cpus {os.cpu_count()} | work {work} ({shutil.disk_usage(work).free / 1e9:.0f} GB free)")
    if shutil.which("nvidia-smi"):
        run("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader")
    try:
        with open("/proc/meminfo") as f:
            log("RAM " + f.readline().strip())
    except OSError:
        pass


def heartbeat(work: Path, every: int, stop: threading.Event) -> None:
    """Print GPU utilisation / memory, RAM and free disk every `every` seconds until stopped."""
    while not stop.wait(every):
        parts = []
        if shutil.which("nvidia-smi"):
            r = subprocess.run("nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits",
                               shell=True, capture_output=True, text=True)
            if r.returncode == 0 and r.stdout.strip():
                u, m, t = [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")]
                parts.append(f"gpu {u}% {float(m) / 1024:.1f}/{float(t) / 1024:.1f} GB")
        try:
            mem = {ln.split(":")[0]: int(ln.split()[1]) for ln in open("/proc/meminfo") if ln.startswith(("MemTotal", "MemAvailable"))}
            parts.append(f"ram {(mem['MemTotal'] - mem['MemAvailable']) / 1e6:.1f}/{mem['MemTotal'] / 1e6:.1f} GB")
        except (OSError, KeyError, ValueError):
            pass
        parts.append(f"disk free {shutil.disk_usage(work).free / 1e9:.0f} GB")
        log("HEARTBEAT " + " | ".join(parts))


def s3_sync(src: str, dst: str, delete: bool) -> int:
    """aws s3 sync with the rebuildable-file excludes (the AWS CLI is installed by install() when the image lacks it)."""
    ex = " ".join(f"--exclude '{e}'" for e in SYNC_EXCLUDE)
    return run(f"aws s3 sync '{src}' '{dst}' {ex} --only-show-errors {'--delete' if delete else ''}".strip())


def stage_checkpoint_files(work: Path, since: float) -> list[Path]:
    """Checkpoint files a stage produced: under CKPT_INCLUDE, not CKPT_EXCLUDE, modified at or after the stage started."""
    from fnmatch import fnmatch

    out = []
    for p in sorted(work.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(work).as_posix()
        if any(fnmatch(rel, pat) for pat in CKPT_INCLUDE) and not any(fnmatch(rel, pat) for pat in CKPT_EXCLUDE) \
                and p.stat().st_mtime >= since:
            out.append(p)
    return out


def stage_manifest(work: Path, files: list[Path], stage: str, started: float, finished: float) -> dict:
    """What a checkpoint holds: stage, job, code version, timing, files, and the holdout reports written by the stage."""
    import json

    metrics = {}
    for p in files:
        if p.name in ("holdout.json", "report.json") or p.parent.name == "measure":
            try:
                metrics[p.relative_to(work).as_posix()] = json.loads(p.read_text())
            except (OSError, ValueError):
                pass
    return {"stage": stage, "run_work": os.environ.get("BER_WORK_S3", ""), "job": os.environ.get("BER_JOB_NAME", ""),
            "git": os.environ.get("BER_GIT_SHA", ""), "started": datetime.fromtimestamp(started).isoformat(timespec="seconds"),
            "finished": datetime.fromtimestamp(finished).isoformat(timespec="seconds"), "minutes": round((finished - started) / 60, 1),
            "files": [{"path": p.relative_to(work).as_posix(), "bytes": p.stat().st_size} for p in files], "metrics": metrics}


def save_checkpoint(work: Path, stage: str, index: int, started: float, ckpt_s3: str) -> None:
    """Freeze the stage's checkpoint files into their own, never-overwritten S3 folder <ckpt_s3>/<NN>-<stage>-<time>/."""
    import json

    files = stage_checkpoint_files(work, started)
    snap = work / ".ckpt_stage"
    shutil.rmtree(snap, ignore_errors=True)
    for p in files:  # hard links: no copy of the data, same filesystem
        dst = snap / p.relative_to(work)
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.link(p, dst)
    snap.mkdir(parents=True, exist_ok=True)
    (snap / "manifest.json").write_text(json.dumps(stage_manifest(work, files, stage, started, time.time()), indent=2, default=str))
    folder = f"{index:02d}-{stage}-{datetime.now():%Y%m%d-%H%M}/"
    dest = f"{ckpt_s3.rstrip('/')}/{folder}"
    # bucket-owner-full-control: the common bucket belongs to a teammate's account, whose members must be able to read the files
    rc = run(f"aws s3 sync '{snap}' '{dest}' --acl bucket-owner-full-control --only-show-errors")
    fallback = os.environ.get("BER_CKPT_FALLBACK", "").rstrip("/")
    if rc != 0 and fallback:  # the common bucket refused the job's role: keep the checkpoint in the own bucket instead
        log(f"checkpoint upload to {dest} failed (exit {rc}); using the fallback {fallback}/")
        dest = f"{fallback}/{folder}"
        rc = run(f"aws s3 sync '{snap}' '{dest}' --only-show-errors")
    shutil.rmtree(snap, ignore_errors=True)
    log(f"CHECKPOINT {stage}: {len(files)} files, {sum(p.stat().st_size for p in files) / 1e6:.0f} MB -> {dest}"
        + ("" if rc == 0 else f" (upload FAILED, exit {rc})"))


def install() -> None:
    if os.environ.get("BER_SKIP_INSTALL") == "1":
        return
    reqs = " ".join(f"-r {CODE / r}" for r in ("requirements.txt", "requirements-gpu.txt") if (CODE / r).exists())
    if run(f"{sys.executable} -m pip install -q --no-cache-dir {reqs}") != 0:
        raise SystemExit("pip install of the pinned requirements failed")
    if not shutil.which("aws"):
        run(f"{sys.executable} -m pip install -q --no-cache-dir awscli")
    if not shutil.which("make"):
        run("apt-get update -qq && apt-get install -y -qq make")


def fail(msg: str) -> None:
    log(f"FAILED: {msg}")
    out = ML / "output"
    out.mkdir(parents=True, exist_ok=True)
    (out / "failure").write_text(msg[:1000])
    sys.exit(1)


def main() -> None:
    t0 = time.time()
    stages = os.environ.get("BER_STAGES", "smoke").split()
    work_s3 = os.environ.get("BER_WORK_S3", "").rstrip("/")
    work = pick_work_dir()
    log(f"stages: {stages} | work: {work} | work on S3: {work_s3 or '(none)'} | code: {CODE}")
    machine_summary(work)
    install()
    if work_s3 and s3_sync(work_s3, str(work), delete=False) != 0:
        fail(f"could not restore WORK from {work_s3}")
    official = DATA / "utils" / "validate_submission.py"
    if official.exists():  # predict.emit runs the organisers' validator when it finds it here
        (work / "official").mkdir(exist_ok=True)
        shutil.copy(official, work / "official" / "validate_submission.py")
    env = {**os.environ, "BER_DATA": str(DATA), "BER_WORK": str(work), "PYTHONPATH": str(CODE / "src"), "PYTHONUNBUFFERED": "1"}
    stop = threading.Event()
    threading.Thread(target=heartbeat, args=(work, int(os.environ.get("BER_HEARTBEAT", "120")), stop), daemon=True).start()
    make = os.environ.get("BER_MAKE_CMD", "make")
    py = sys.executable
    ckpt_s3 = os.environ.get("BER_CKPT_S3", "").rstrip("/")
    base = int(os.environ.get("BER_CKPT_BASE", "0"))  # numbering continues across the jobs of a run (set by sm.py)
    for i, stage in enumerate(stages, start=1):
        ts = time.time() - 1  # 1 s slack: file mtimes can round down
        log(f"========== STAGE {stage}: started")
        if stage == "smoke":
            rc = run(f"{py} -c \"import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), "
                     f"torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')\"", env=env)
            rc = rc or run(f"{py} -m pytest -q -p no:warnings src/tests", cwd=CODE, env=env)
        else:
            rc = run(f"{make} {stage} PY={py} TPY={py}", cwd=CODE, env=env)
        log(f"========== STAGE {stage}: {'OK' if rc == 0 else f'FAILED (exit {rc})'} in {(time.time() - ts) / 60:.1f} min")
        if rc == 0 and ckpt_s3 and stage != "smoke":
            save_checkpoint(work, stage, base + i, ts, ckpt_s3)
        if work_s3:
            s3_sync(str(work), work_s3, delete=True)
            log(f"WORK saved to {work_s3}")
        if rc != 0:
            stop.set()
            fail(f"stage {stage} exited with {rc}; finished stages are saved in {work_s3}")
    stop.set()
    log(f"ALL STAGES DONE in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
