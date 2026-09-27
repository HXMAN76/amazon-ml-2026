# Qwen experiment context and handoff

Last updated: 27 September 2026 (IST)

## 1. Objective

This experiment adds a model-family-diverse cross-encoder to the business-entity-resolution pipeline. The existing first-stage and reranking system already uses lexical, dense, contextual, and multilingual E5-derived signals. The Qwen model is not intended to replace those models or emit a submission by itself. Its score becomes an additional feature in the final contextual stack (`s24`).

The intended gain is improved discrimination inside the ambiguous candidate band, especially where the existing first-stage model is uncertain or overconfident. The experiment is accepted only if `s24` beats `s22` on the same protected 150,000-S1 holdout with a positive paired 95% confidence interval and no material US/India regression.

## 2. Architecture

The Qwen component is a binary pair cross-encoder:

- Base family: `Qwen3-0.6B`.
- Licence: Apache-2.0.
- Size: approximately 0.6 billion parameters, below the competition's 8B limit.
- Input prompt:

  ```text
  Same business?
  A: <source-1 record text>
  B: <candidate record text><|im_end|>
  ```

- Both entity records are jointly encoded, allowing attention across their names, addresses, and other rendered fields.
- Output: one binary-match logit converted to probability `xs`.
- Maximum sequence length: 192 tokens.
- Inference dtype: BF16 on NVIDIA A10G.
- Scoring universe: only the existing `v7` ambiguous band, not every possible record pair.

The downstream `s24` stack combines Qwen's `xs3` feature with:

- the `v7` first-stage probability and engineered pair features;
- the existing cross-encoder score directories `xenc2F_v7` and `xenc_v7`;
- cross-encoder consensus and competition features;
- TF-IDF and contextual/list-level features;
- decoy and extra structural features.

This model diversity is important: a Qwen decoder-family cross-encoder can make different errors from the encoder-only E5 models, while the stack learns when each signal is trustworthy.

## 3. Source and immutable inputs

The implementation was taken from the latest `origin/sai` state used for this experiment:

```text
3977dc5  Qwen scoring and stack job templates, handoff section 9
```

Relevant source files:

- `src/ber/stages/xenc.py`
- `src/ber/stages/stack.py`
- `aws/jobs/qwen_score.sh`
- `aws/jobs/qwen_stack.sh`

The fitted Qwen checkpoint and pair lists came from the completed team export:

```text
work/xenc3Q/model/
work/xenc3Q_v7/train.parquet
work/xenc3Q_v7/test.parquet
```

The checkpoint was already fine-tuned before this continuation. It was not retrained on the Mumbai notebook.

Documented fit recipe for the saved checkpoint:

- approximately 966k training pairs;
- one epoch;
- BF16;
- gradient checkpointing;
- binary classification with the prompt shown above.

## 4. Mumbai execution environment

Notebook/container:

```text
SageMaker notebook: sai-india
AWS region: ap-south-1
Container host observed during scoring: algo-1-yr5q3
GPU: NVIDIA A10G, 23,028 MiB VRAM
System RAM: approximately 30 GiB
Notebook EBS: approximately 98 GiB
```

Experiment root:

```text
/opt/ml/input/data/notebook/qwen-exp
```

Important paths:

```text
Code:       /opt/ml/input/data/notebook/qwen-exp/ber
Environment:/opt/ml/input/data/notebook/qwen-exp/env
Work:       /opt/ml/input/data/notebook/qwen-exp/work
Model:      /opt/ml/input/data/notebook/qwen-exp/work/xenc3Q/model
Pair lists: /opt/ml/input/data/notebook/qwen-exp/work/xenc3Q_v7
Logs:       /opt/ml/input/data/notebook/qwen-exp/logs
State:      /opt/ml/input/data/notebook/qwen-exp/state
```

An isolated Python 3.12 environment was created on notebook EBS because the container's system Python 3.8 environment had Torch 1.9.1 and no compatible Transformers installation.

Key runtime versions:

```text
Python       3.12
Torch        2.9.1+cu128
Transformers 5.17.0
Polars       1.44.2
```

Transformers 5.17.0 is significant: the saved checkpoint's `config.json` records that version, and Transformers 4.57.6 failed to load its tokenizer because of the Qwen `extra_special_tokens` representation. `protobuf` was also installed.

The inherited container set Python 3.8 paths through `PYTHONPATH`. Commands therefore cleared `PYTHONHOME` and `PYTHONPATH` before invoking the isolated environment, then explicitly set `PYTHONPATH` to the BER source directory.

## 5. Input sizes

The restored scoring inputs were validated with Polars:

| Split | Candidate pairs |
|---|---:|
| Test | 2,459,024 |
| Train | 2,129,577 |
| Total | 4,588,601 |

These are ambiguous-band pairs. The Qwen model was not run over the complete combinatorial record universe.

## 6. Smoke testing and throughput tuning

Before production scoring, an isolated smoke directory was created using the real test input and a separate output path. The first successful smoke test scored 4,096 pairs with:

```text
score_batch=256
max_len=192
```

It produced exactly 4,096 finite `xs` values with columns `q`, `pid`, and `xs`. The observed probability range was approximately `1.1e-9` to `0.999998`.

Batch-size benchmarks on the A10G showed:

| Batch size | Observed result |
|---:|---|
| 256 | about 171 pairs/second; unnecessarily conservative |
| 1024 | about 256 pairs/second; fastest tested stable setting |
| 2048 | slightly slower than 1024 |

Production scoring therefore used:

```text
score_batch=1024,max_len=192
```

During scoring, the GPU was normally near 100% compute utilization. VRAM use was only around 5-6 GiB; this is expected for a 0.6B model and does not indicate an idle GPU.

## 7. Production queue

The production queue ran the splits sequentially on the single A10G:

1. score the test ambiguous-band pairs;
2. validate test input/output row equality;
3. hash and upload the test scores;
4. score the train ambiguous-band pairs;
5. validate train input/output row equality;
6. hash and upload the train scores.

Queue script on the notebook:

```text
/opt/ml/input/data/notebook/qwen-exp/qwen_score_queue.sh
```

Log:

```text
/opt/ml/input/data/notebook/qwen-exp/logs/qwen-score-queue.log
```

The queue used completion markers under `state/` and separate local-scoring and upload completion states so an S3 upload failure would not require rescoring millions of pairs.

## 8. Completed results

The complete queue finished successfully:

```text
DONE test  2026-09-26T21:08:42+00:00
DONE train 2026-09-26T23:12:21+00:00
QUEUE COMPLETE
```

Measured runtimes:

| Split | Pairs | Runtime |
|---|---:|---:|
| Test | 2,459,024 | about 2 h 31 min |
| Train | 2,129,577 | 7,411 s, about 2 h 04 min |

Training-band diagnostic:

```text
Qwen xs average precision: 0.9795
Existing p1 average precision: 0.9177
Absolute AP improvement:      +0.0618
```

This is a strong pair-ranking result inside the ambiguous band. It is **not** the competition F0.5 and must not be reported as the final model score. Only the protected `s24` holdout evaluation can determine whether this signal improves entity-level matching.

Local completed outputs:

```text
/opt/ml/input/data/notebook/qwen-exp/work/xenc3Q_v7/test_xs.parquet
/opt/ml/input/data/notebook/qwen-exp/work/xenc3Q_v7/train_xs.parquet
```

Durable S3 copies:

```text
s3://sagemaker-ap-south-1-767397931665/qwen-s24-3977dc5/test_xs.parquet
s3://sagemaker-ap-south-1-767397931665/qwen-s24-3977dc5/test_xs.parquet.sha256
s3://sagemaker-ap-south-1-767397931665/qwen-s24-3977dc5/train_xs.parquet
s3://sagemaker-ap-south-1-767397931665/qwen-s24-3977dc5/train_xs.parquet.sha256
```

No AWS access keys or other credentials should be written into this document, scripts, logs, or repository.

## 9. What remains

### 9.1 Restore the full stack workspace on a high-memory machine

The Qwen scoring phase is complete, but the `s24` contextual stack has not yet been built or trained. The stack requires:

- the complete exported `work` directory;
- `v7` first-stage artifacts;
- `xenc2F_v7` and `xenc_v7` score artifacts;
- `work/models/s22` including its protected holdout predictions and metadata;
- the two completed Qwen score files under `work/xenc3Q_v7/`;
- approximately 128 GiB system RAM.

The Mumbai notebook has only about 30 GiB RAM. It is suitable for GPU scoring but not for the documented full `s24` build. The next machine should prioritize RAM; a GPU is not required for the XGBoost stack stage.

### 9.2 Build, train, predict, and compare `s24`

From the restored BER environment:

```bash
export BER_DATA=/path/to/dataset
export BER_WORK=/path/to/work
export PYTHONPATH=/path/to/ber/src

A="--base v7 --tag _q --xenc --xcons \
--xenc-fit-more 300000 --decoy --extra --sub-q 1500000 \
--xenc-dir xenc2F_v7 \
--xenc-dir2 xenc_v7 \
--xenc-dir3 xenc3Q_v7"

python -m ber.stages.stack build --split train $A
python -m ber.stages.stack tfidf --split train --tag _q
python -m ber.stages.stack build --split test $A
python -m ber.stages.stack tfidf --split test --tag _q
python -m ber.stages.stack train \
  --name s24 --base v7 --tag _q \
  --set max_depth=9,eta=0.05,rounds=1500,early_stop=50
python -m ber.stages.stack predict --name s24
python src/scripts/check_submission.py "$BER_WORK/output/s24" "$BER_DATA/test"
python src/scripts/paired_models.py s22 s24
```

The checked-in template is `aws/jobs/qwen_stack.sh`; adjust only paths/environment activation for the target machine.

### 9.3 Shipping gate

Promote `s24` only if all of the following pass:

- paired `s24 - s22` delta has a 95% confidence interval whose lower bound is above zero;
- evaluation uses the identical untouched 150k-S1 holdout;
- there is no material US or India regression;
- train and test joins preserve all expected rows and do not duplicate `(q, pid)` pairs;
- Qwen scores remain a separate feature from existing E5 scores;
- ownership, candidate-subset, ID, and row-count validators pass.

If `s24` is neutral or loses, retain `s22` and do not ship Qwen merely because its band AP is high.

### 9.4 France structural decoding

If `s24` wins, transfer the best already-measured France structural policy onto `s24` probabilities rather than retraining Qwen on invented French labels. The current policy family includes:

- `typeswap` structural filtering;
- France-specific thresholding;
- post-exclusivity caps of five S2 and six S3 matches per S1.

The winning combination must be regenerated from `s24`'s `pair_p.parquet`, validated, and compared with the existing rollback submission. If `s24` does not win, keep the strongest validated `s22` France variant.

## 10. Current status summary

| Stage | Status |
|---|---|
| Qwen checkpoint fine-tuning | Complete before Mumbai continuation |
| Restore checkpoint and pair lists | Complete |
| Reproduce compatible runtime | Complete |
| Smoke test and tune batch size | Complete |
| Score test band | Complete and backed up |
| Score train band | Complete and backed up |
| Train `s24` contextual stack | Not yet run |
| Paired holdout F0.5 evaluation | Not yet available |
| Generate/validate `s24` submission | Not yet run |
| Apply winning France decoder | Pending `s24` decision |

The GPU-intensive work is finished. The critical next result is the paired entity-level F0.5 comparison of `s24` against `s22`, not another Qwen scoring run.
