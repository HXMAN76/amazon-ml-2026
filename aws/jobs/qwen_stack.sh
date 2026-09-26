#!/bin/bash
# s24 = s22 plus the Qwen3-0.6B cross-encoder score xs3 (band pairs only). Needs the full work directory (stack features, first stage v7, xenc2F_v7, xenc_v7) and 128 GB RAM.
# Run in the `ber` conda env after qwen_score.sh. Result: work/output/s24 (matching_results.tsv, candidate_pairs.tsv), paired test against s22.
set -ex
source /home/ec2-user/anaconda3/etc/profile.d/conda.sh; conda activate ber
cd /home/ec2-user/SageMaker/ber
export BER_DATA=/home/ec2-user/SageMaker/dataset BER_WORK=/home/ec2-user/SageMaker/work PYTHONPATH=/home/ec2-user/SageMaker/ber/src
A="--base v7 --tag _q --xenc --xcons --xenc-fit-more 300000 --decoy --extra --sub-q 1500000 --xenc-dir xenc2F_v7 --xenc-dir2 xenc_v7 --xenc-dir3 xenc3Q_v7"
python -m ber.stages.stack build --split train $A
python -m ber.stages.stack tfidf --split train --tag _q
python -m ber.stages.stack build --split test $A
python -m ber.stages.stack tfidf --split test --tag _q
python -m ber.stages.stack train --name s24 --base v7 --tag _q --set max_depth=9,eta=0.05,rounds=1500,early_stop=50
python -m ber.stages.stack predict --name s24
python src/scripts/check_submission.py $BER_WORK/output/s24 $BER_DATA/test
python src/scripts/paired_models.py s22 s24   # needs work/models/s22 (holdout_pred.parquet, holdout.json), included in the export
