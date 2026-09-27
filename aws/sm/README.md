# Running the pipeline on a GPU with SageMaker training jobs (no notebook)

`sm.py` (laptop) starts a job on an **ml.g5.4xlarge** (A10G 24 GB, 16 vCPU, 64 GB RAM) that runs `make` stages of
the pipeline at the repository root (`src/`, `configs/`, `Makefile`) with `entry.py`, and streams the job's log to your terminal. You pay only while a job runs.
Every job of a run shares its working directory on S3 (`s3://sagemaker-us-east-1-<account>/ber/work/<run>/`). A job restores it,
runs its stages and saves it after every stage, so the next job continues and a failed job keeps its finished stages.

Use the smssh-venv Python (it has boto3 and `botocore[crt]` for `aws login` sessions); the profile defaults to `barani`.

```bash
alias sm='smssh-venv/bin/python aws/sm/sm.py'
sm check --quota                    # read-only: who am I, bucket, role, dataset access, g5 quotas
sm setup                            # one time: bucket, execution role ber-sagemaker-training, dataset + validator copy
sm submit --stages smoke            # ~10 min: GPU check + the 33 tests on the GPU box (proves the whole chain)
sm submit --run v5 --stages v5_candidates                  # job A: prepare, cascade blocking, dense channels, exact keys
sm submit --run v5 --stages "v5_pass1 v5_expand v5_pass2"  # job B: two first-stage passes with sibling expansion
sm submit --run v5 --stages "v5_xenc v5_stack v5_measure"  # job C: cross-encoder, stacking, decision, measurements
sm pull --run v5 --model s5         # reports + output/s5/matching_results.tsv -> output/sm/v5/
```

## Watching live

- `sm submit` follows the job until it ends and prints:
  - the job's status changes (`[status] Starting`, `Downloading`, `Training`, ...);
  - every log line of the pipeline;
  - `========== STAGE <name>: started / OK in x min` markers;
  - a `HEARTBEAT gpu % | ram | disk` line every 2 minutes (`--heartbeat`);
  - a final line with the status, the billable minutes and the approximate cost.
- **Ctrl-C only detaches**; the job keeps running.
- Reattach from any terminal: `sm logs <job>`. The AWS CLI alternative is
  `aws logs tail /aws/sagemaker/TrainingJobs --log-stream-name-prefix <job> --follow --profile barani`.
- `sm status` lists recent jobs with their runtime; `sm status <job>` shows one job with its cost; `sm stop <job>` stops it.

## Checkpoints of every stage (S3)

After every stage that finishes OK, `entry.py` freezes the files that stage produced into their own folder in the **team's common
bucket**, which is never overwritten. The files carry `bucket-owner-full-control` so teammates in other accounts can read them. If
the common bucket refuses the job's role, the checkpoint goes to `s3://sagemaker-us-east-1-<account>/ber/checkpoints/<run>/`.

```
s3://ml-challenge-nooglers/ml-challenge-2026/checkpoints/<owner = profile name>/<run>/<NN>-<stage>-<yyyymmdd-hhmm>/
    manifest.json          stage, job, git version, start/end, minutes, file list, holdout reports of the stage
    models/<name>/...      xgb.json, config.json, report.json, holdout.json, oof / holdout predictions
    dense*/model_ft/...    fine-tuned encoders        xenc/model/...  fine-tuned cross-encoder
    measure/*.json         output/<name>/matching_results.tsv   keys|expand/*/pairs.parquet
```

- Rebuildable bulk (candidate shards, feature parts, `p1_rest`) stays only in the WORK mirror.
- Numbering continues across the jobs of a run.
- `sm checkpoints --run v5` lists them with their scores; `sm pull --run v5 --checkpoint <id>` downloads one.
- The WORK mirror (`ber/work/<run>/`) is what the next job resumes from. The checkpoints are the history you can always go back to.

## Notes

- The image is the AWS PyTorch training DLC `pytorch-training:2.7.1-gpu-py312`: CUDA matched to the host, Python 3.12 for the pins.
  `entry.py` installs `requirements.txt` + `requirements-gpu.txt` on top. The models (multilingual-e5-small, the cross-encoder) download
  from Hugging Face inside the job, and the job's quota allows one ml.g5.4xlarge at a time.
- Rebuildable large files (DuckDB pool indexes, the raw K-150 candidate lists) are not saved to S3.
- If a job fails, fix the code and resubmit the same `--run` with the remaining stages: `make` skips stages whose outputs and
  parameters are unchanged only where the Makefile has stamps (prepare, sample); otherwise pass just the stages still to run.
- Tests: `python -m pytest -q aws/sm/test_sm.py` (fake AWS clients, no account needed).
