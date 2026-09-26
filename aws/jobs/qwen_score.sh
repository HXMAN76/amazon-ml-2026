#!/bin/bash
# Score the band pairs with the fitted Qwen3-0.6B cross-encoder (work/xenc3Q/model). Needs work/xenc3Q_v7/{train,test}.parquet (pair lists with text) and the
# repo code; 1 GPU: run the two splits one after the other, several GPUs: one split per GPU. About 200 pairs/s per A10G. Run in the `pytorch` conda env.
# Replace <BUCKET> by your bucket (or sync the code by git pull). Output: work/xenc3Q_v7/{train,test}_xs.parquet
set -ex
source /home/ec2-user/anaconda3/etc/profile.d/conda.sh; conda activate pytorch
cd /home/ec2-user/SageMaker/ber
export BER_DATA=/home/ec2-user/SageMaker/dataset BER_WORK=/home/ec2-user/SageMaker/work PYTHONPATH=/home/ec2-user/SageMaker/ber/src
pip install -q transformers sentencepiece polars==1.44.2 duckdb==1.5.5 pyyaml scikit-learn xgboost
CUDA_VISIBLE_DEVICES=0 python -m ber.stages.xenc score --split train --dir xenc3Q_v7 --model-dir xenc3Q/model --set score_batch=256,max_len=192 &
P1=$!
CUDA_VISIBLE_DEVICES=${SECOND_GPU:-0} python -m ber.stages.xenc score --split test --dir xenc3Q_v7 --model-dir xenc3Q/model --set score_batch=256,max_len=192 &
P2=$!
wait $P1; wait $P2
# with one GPU set SECOND_GPU=0 and expect the two processes to share it (slower); or run the test split first, it is the one that matters for the France rules
