"""Run tracking: MLflow (sqlite under WORK/mlflow.db) plus an append-only runs.jsonl as backup."""

from __future__ import annotations

import json
import os
import subprocess
import time

from ber.config import paths

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


def _git_sha() -> str:
    if os.environ.get("BER_GIT_SHA"):
        return os.environ["BER_GIT_SHA"]
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:  # noqa: BLE001 - not a git checkout on the notebook
        return "nogit"


def log_stage(stage: str, params: dict, metrics: dict[str, float], extra: dict | None = None) -> None:
    """Append a stage record to runs.jsonl and log it to MLflow (never fails the stage)."""
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "stage": stage, "git": _git_sha(), "params": params,
           "metrics": metrics, **(extra or {})}
    runs = paths()["runs"]
    runs.mkdir(parents=True, exist_ok=True)
    with (runs / "runs.jsonl").open("a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    try:
        import mlflow

        mlflow.set_tracking_uri(f"sqlite:///{paths()['work'].resolve() / 'mlflow.db'}")
        mlflow.set_experiment("ber")
        with mlflow.start_run(run_name=stage):
            mlflow.set_tag("git_sha", rec["git"])
            mlflow.log_params({k: str(v)[:250] for k, v in _flat(params).items()})
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
    except Exception as e:  # noqa: BLE001 - tracking must never break a stage
        print(f"[tracking] mlflow skipped: {e}", flush=True)


def _flat(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        out.update(_flat(v, key + ".") if isinstance(v, dict) else {key: v})
    return out
