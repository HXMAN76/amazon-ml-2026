"""Experiment tracking: MLflow (local sqlite) when installed + always an append-only runs.jsonl.

runs.jsonl is the team-wide leaderboard: sync it to s3://<bucket>/07-experiments/<name>/ so
everyone sees every experiment regardless of which machine/account ran it.

    with track("EXP-024", params={...}, tags={"dataset": "v1", "features": "v3"}) as run:
        ...
        run.log_metrics({"cv_smape": 43.1})
"""

from __future__ import annotations

import contextlib
import json
import os
import platform
import socket
import subprocess
import time
from pathlib import Path

RUNS_FILE = Path(os.environ.get("AMLC_RUNS_FILE", "artifacts/runs.jsonl"))


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:  # noqa: BLE001
        return ""


def environment() -> dict:
    env = {
        "git_commit": _git("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "host": socket.gethostname(),
        "member": os.environ.get("AMLC_MEMBER", ""),
        "python": platform.python_version(),
    }
    try:
        import torch

        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        pass
    return env


class Run:
    def __init__(self, name: str, params: dict, tags: dict):
        self.record = {"name": name, "start": time.strftime("%Y-%m-%dT%H:%M:%S"), "params": params,
                       "tags": {**environment(), **tags}, "metrics": {}}
        self._mlflow = None

    def log_metrics(self, metrics: dict) -> None:
        self.record["metrics"].update(metrics)
        if self._mlflow:
            self._mlflow.log_metrics({k: float(v) for k, v in metrics.items()})

    def log_artifact(self, path: str) -> None:
        self.record.setdefault("artifacts", []).append(path)
        if self._mlflow:
            self._mlflow.log_artifact(path)


@contextlib.contextmanager
def track(name: str, params: dict | None = None, tags: dict | None = None, experiment: str = "amlc-2026"):
    run = Run(name, params or {}, tags or {})
    mlflow_ctx = contextlib.nullcontext()
    try:
        import mlflow

        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
        mlflow.set_experiment(experiment)
        mlflow_ctx = mlflow.start_run(run_name=name)
        run._mlflow = mlflow
    except ImportError:
        pass
    t0 = time.time()
    with mlflow_ctx:
        if run._mlflow:
            run._mlflow.log_params({k: str(v)[:500] for k, v in run.record["params"].items()})
            run._mlflow.set_tags({k: str(v) for k, v in run.record["tags"].items()})
        try:
            yield run
            run.record["status"] = "ok"
        except BaseException:
            run.record["status"] = "failed"
            raise
        finally:
            run.record["seconds"] = round(time.time() - t0, 1)
            RUNS_FILE.parent.mkdir(parents=True, exist_ok=True)
            with RUNS_FILE.open("a") as f:
                f.write(json.dumps(run.record, default=str) + "\n")
