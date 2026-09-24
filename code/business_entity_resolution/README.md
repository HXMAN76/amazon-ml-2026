# Business Entity Resolution: pipeline

Stage pipeline driven by a Makefile with hash-based caching. Every tunable is in `configs/params.yaml`.
Design rationale and the approved plan are in the repository root: `plan.md`, `research.md`, `context.md`.
No external lookups (APIs, registries, geocoding) are used anywhere; pretrained models, when added, are MIT or
Apache-2.0 and at most 8B parameters.

## Stages (implemented so far)

| Stage | Command | Output (under `$BER_WORK`) |
|---|---|---|
| prepare | `make prepare` | `parquet/{train,test}/source{1,2,3}.parquet`, `parquet/train/labels.parquet` |
| sample | `make sample` | `sample/train_s1.parquet` (250k S1 queries with CV folds) |
| block (train sample / test) | `python -m ber.stages.block --split train` / `--split test` | `blocks/{split}/cand_*.parquet` |
| block_eval | `make block_eval` | `blocks/eval_report.json` (recall and miss breakdown per configuration) |

Features, training, calibration, decision and prediction stages follow in later phases.

## Run

```bash
pip install -r requirements.txt
export BER_DATA=/path/to/dataset          # contains train/ and test/ TSV folders
export BER_WORK=/path/to/work             # outputs
make prepare sample block_eval
make test
```

`make` reruns a stage only when its params section or source files changed (stamps under `$BER_WORK/.stamps`).
Runs are logged to MLflow (sqlite, `$BER_WORK/mlflow.db`) and `$BER_WORK/runs/runs.jsonl`.

## Legacy v0

`ber.run`, `blocking.py`, `features.py`, `model.py`, `decide.py`, `normalize.py`, `data.py` and `validate.py` are the
first baseline (dense per-country TF-IDF kNN, 49 pair features, LightGBM). It was only tested on synthetic data and
does not scale to the real pool; it is kept until the new stages replace it.
