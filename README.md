# Amazon ML Challenge 2026: team pipeline

The challenge window runs from **24 Sep 18:30 UTC to 27 Sep 18:29 UTC**, which is 25 Sep 00:00 IST to 27 Sep 23:59 IST.
Ranking uses the best score. Ties go to whoever submitted earlier, so a valid baseline submitted early is worth a lot.
Before submitting you also need a 1–2 page approach document and a zip of the code.

Rule of the stack: **cheap first, GPU only when needed, and every expensive output cached and sharded.**

```
raw csv ──> downloader (async, resumable) ──> images + manifest ──> attach-images ──> *_img.parquet
                                                                              │
             text ──> tfidf/svd, numeric stats, sentence-embeddings ──┐       ├─> image embeddings (SigLIP/CLIP)
                                                                      ├──> GBM k-fold (OOF) ──> blend ──> post-process ──> validator ──> submission.csv
             VLM (4-bit, sharded, chunk-checkpointed) ─> parsed values ┘
```

Compute pools, in order of use:
1. **Local RTX 4060 (8 GB)**: embeddings, small VLMs in 4-bit, GBMs on 24 CPU cores.
2. **Kaggle**: about 30 free GPU-hours a week per account on T4×2 or P100, so roughly 120 hours for the team. Use [notebooks/kaggle_bootstrap.py](notebooks/kaggle_bootstrap.py).
3. **AWS g5/g6**: only if quota was approved and the account is on the Paid plan. See [aws/40_launch_gpu.sh](aws/40_launch_gpu.sh). The instance auto-terminates after `MAX_HOURS`.

## Before the dataset drops (each member)

- [ ] Accept the GitHub invite and clone. Run `make setup`, then `make test`.
- [ ] **Kaggle**: create an account, **verify your phone** (needed for GPU and internet), and add Secrets: `GITHUB_TOKEN`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AMLC_BUCKET`.
- [ ] **Hugging Face**: create an account and a read token, then run `huggingface-cli login`. Some models (Llama, Gemma, PaliGemma) need you to accept a license on the model page first.
- [ ] **AWS**: work in your own account's CloudShell (console, top bar `>_`), region **ap-south-1**:
  ```bash
  git clone https://github.com/HXMAN76/amazon-ml-2026.git && cd amazon-ml-2026
  bash aws/00_account_check.sh                    # paste output in team chat
  ALERT_EMAIL=you@x.com bash aws/02_budget.sh     # cost alarm
  bash aws/01_request_quotas.sh                   # GPU quotas (Paid plan only, see below)
  bash aws/20_compute_role.sh                     # role for EC2/SageMaker (needs env.sh filled)
  ```
- [ ] **Account A only**: fill the account ids in [aws/env.sh](aws/env.sh) and commit. Then run `bash aws/10_hub_bucket.sh` and `bash aws/15_external_user.sh`.
- [ ] **B, C, D**: run `bash aws/30_verify_hub_access.sh`. It should print PASS for list, write, read, and the 00-raw denial.
- [ ] **Laptop AWS access**: run `aws login --profile amlc`, then `export AWS_PROFILE=amlc AMLC_BUCKET=amlc-2026-hub-<ACCOUNT_A>`.

### Free plan vs Paid plan

Free plan accounts **cannot launch GPU instances**, and GPU quota requests are usually denied.
Upgrading to Paid (Billing console > Free Tier > Upgrade) keeps the remaining credits.
After upgrading, usage beyond the credits is billed to your card. The budget from `02_budget.sh` counts usage *before* credits are applied, so its emails show real burn.
Recommendation: upgrade one or two accounts at most. The others stay Free plan and are used for S3 and CPU work.
**Do not join AWS Organizations.** Doing so forces the Paid plan on every account and merges Free Tier benefits.

## Hour 0–3 playbook

1. **Read the rules twice.** Note the allowed model params and license (2025 capped this), whether external data is allowed, the metric, the submission format, and the daily submission limit.
2. Fill in [configs/submission.yaml](configs/submission.yaml) and add the metric to [src/amlc/evaluation.py](src/amlc/evaluation.py) if it is new.
3. Upload the raw dataset once: `aws s3 sync data/raw/ s3://$AMLC_BUCKET/00-raw/`. This folder is immutable.
4. Split image downloads across the four machines using row ranges. Example for the test set with N rows:
   ```bash
   python -m amlc.data.downloader --input data/raw/test.csv --url-col image_link \
       --out data/images --max-side 512 --start 0 --end 40000        # member 1: 0-40k, member 2: 40k-80k ...
   make push-images                                                  # everyone syncs to 01-images/
   ```
5. **Submit a baseline within about 2 hours.** Use TF-IDF plus a GBM for tabular/text tasks, or a regex/unit parser for extraction tasks. That gives real leaderboard feedback and a timestamp.
6. Then run the parallel tracks. Suggested split:

| Member | Track |
|---|---|
| A | data hub, downloads, validator, submissions, write-up |
| B | text features, GBMs, CV, blending |
| C | image embeddings (SigLIP/CLIP), MLP heads |
| D | VLM / LoRA fine-tune (quantized), post-processing |

## Commands

```bash
# attach local image paths (after download)
python -m amlc.data.io --input data/raw/train.csv --images data/images --out data/train_img.parquet

# image / text embeddings, sharded (--shard i --num-shards n), resumable
python -m amlc.features.embed image --input data/train_img.parquet --col image_path \
    --model google/siglip2-base-patch16-224 --out artifacts/emb/train/siglip2 --shard 0 --num-shards 1
python -m amlc.features.embed text --input data/raw/train.csv --col catalog_content \
    --model BAAI/bge-small-en-v1.5 --out artifacts/emb/train/bge-small
python -m amlc.inference.shard artifacts/emb/train/siglip2 --to artifacts/emb/train/siglip2.parquet --expected-rows N

# VLM, 4-bit, chunk-checkpointed, prompt filled from row columns
python -m amlc.inference.vlm --input data/test_img.parquet --model Qwen/Qwen2.5-VL-3B-Instruct --load-in-4bit \
    --prompt "What is the {entity_name}? Answer '<number> <unit>' only." --out artifacts/vlm/test/q3b --shard 0 --num-shards 4

# LoRA fine-tune (auto-resumes from latest checkpoint)
python -m amlc.training.finetune_text --train data/raw/train.csv --text-col catalog_content --label-col price \
    --task reg --target log1p --model microsoft/deberta-v3-small --output-dir checkpoints/deb-lora --bf16

# validate before EVERY upload
python -m amlc.submission.validator outputs/submission.csv --test data/raw/test.csv

make mlflow        # experiment UI (local sqlite)
make push-artifacts / pull-artifacts
bash aws/99_stop_all.sh   # cost panic button
```

GBM plus blend, used from a notebook:

```python
from amlc.models.gbm import cv_train, blend
from amlc.tracking import track
with track("EXP-003", params={"feats": "tfidf+siglip"}, tags={"dataset": "v1"}) as run:
    r = cv_train(X, y, X_test, model="lgbm", metric="smape", target="log1p", name="EXP-003")
    run.log_metrics({"cv_smape": r["score"]})
```

## Cost rules

1. No GPU runs while nobody is watching it. `40_launch_gpu.sh` always sets an auto-terminate.
2. Use CPU (laptop or Kaggle CPU) for EDA, TF-IDF and GBMs.
3. Cache every expensive output (images, embeddings, VLM answers) in S3 once, and let everyone reuse it.
4. Prefer spot instances. Chunk checkpoints mean an interruption loses at most 500 rows.
5. Run `aws/99_stop_all.sh` before sleeping.
