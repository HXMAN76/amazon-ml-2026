#!/bin/bash
# Reproduces the final submission (model s15) from the raw TSV files. Run from code/business_entity_resolution.
#   BER_DATA=<folder with train/ and test/>  BER_WORK=<scratch, about 120 GB>  bash reproduce_final.sh
# Two Python environments are used: `ber` (Python 3.12, requirements.txt) for everything except the steps marked [torch], which need the
# `pytorch` environment (requirements-gpu.txt: torch with CUDA, transformers, sentencepiece, plus polars, duckdb, pyyaml, scikit-learn).
# Expected time on a 64 vCPU + A10G machine: about 6 hours. Every stage logs to $BER_WORK/runs/runs.jsonl.
set -euxo pipefail
export PYTHONPATH=src
py()  { python -m "$@"; }                 # ber environment
pyt() { "${TORCH_PYTHON:-python}" -m "$@"; }   # [torch] steps: set TORCH_PYTHON to the interpreter of the pytorch environment

# 1. prepare and sample (text normalisation, 250k-S1 sample with folds)
py ber.stages.prepare
py ber.stages.sample

# 2. token blocking: 100 raw candidates per S1 for test and all train S1, learned pruner keeps the best 30
py ber.stages.block --split test --k 100 --out-name test_raw
py ber.stages.block --split train --all-train --k 100 --out-name train_raw
py ber.stages.prune --train
py ber.stages.prune --apply train test

# 3. dense channel for non-Latin names (top 5 per pool record), merged into the candidate shards
pyt ber.stages.dense finetune
pyt ber.stages.dense embed --split train
pyt ber.stages.dense embed --split test
pyt ber.stages.dense retrieve --split train
pyt ber.stages.dense retrieve --split test
py ber.stages.dense merge --split test
py ber.stages.dense merge --split train

# 4. dense channel over name plus address for every record (top 1 per pool record), merged into the shards
pyt ber.stages.dense_all finetune
pyt ber.stages.dense_all embed --split train
pyt ber.stages.dense_all embed --split test
pyt ber.stages.dense_all retrieve --split train
pyt ber.stages.dense_all retrieve --split test
py ber.stages.dense_all merge --split test
py ber.stages.dense_all merge --split train

# 5. first stage v7: features, XGBoost on 850k S1 (sample + 600k), scores for the rest of train (holdout report) and test
py ber.stages.pairs --split train
py ber.stages.pairs --split train --rest
py ber.stages.pairs --split test
py ber.stages.train_gpu --name v7 --extra 600000 --set max_depth=9,eta=0.05,rounds=3000,early_stop=50
py ber.stages.score_rest --name v7
py ber.stages.predict --name v7

# 6. cross-encoder on the uncertain band of v7 (data in `ber`, fitting and scoring in `pytorch`)
py ber.stages.xenc data --base v7 --dir xenc_v7
pyt ber.stages.xenc train --dir xenc_v7 --model-dir xenc_v7/model
pyt ber.stages.xenc score --split train --dir xenc_v7 --model-dir xenc_v7/model
pyt ber.stages.xenc score --split test --dir xenc_v7 --model-dir xenc_v7/model

# 7. stack s15 (shortlist, consensus, digit, TF-IDF, decoy edit and carried features, cross-encoder score), decision and outputs
py ber.stages.stack build --split train --base v7 --tag _i --xenc --xenc-dir xenc_v7 --decoy --extra --sub-q 1500000
py ber.stages.stack tfidf --split train --tag _i
py ber.stages.stack build --split test --base v7 --tag _i --xenc --xenc-dir xenc_v7 --decoy --extra
py ber.stages.stack tfidf --split test --tag _i
py ber.stages.stack train --name s15 --base v7 --tag _i --set max_depth=9,eta=0.05,rounds=1500,early_stop=50
py ber.stages.stack predict --name s15

# 8. checks (the official validator is also run by `predict`/`stack predict` when work/official/validate_submission.py exists)
python src/scripts/check_submission.py "$BER_WORK/output/s15" "$BER_DATA/test"
echo "outputs: $BER_WORK/output/s15/matching_results.tsv and candidate_pairs.tsv"
