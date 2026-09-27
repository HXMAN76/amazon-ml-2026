set -ex
W=/home/ec2-user/SageMaker/work; X=s3://ml-challenge-nooglers/ml-challenge-2026/handoff-nooglers-20260926/work/s29x
aws s3 cp $W/output/s29/pair_p.parquet $X/pair_p.parquet --only-show-errors
aws s3 cp $W/models/s29/config.json $X/config.json --only-show-errors
for f in matching_results.tsv candidate_pairs.tsv; do aws s3 cp $W/output/v8w_s29_FIN/$f $X/FIN/$f --only-show-errors; done
echo READY > /tmp/r && aws s3 cp /tmp/r $X/READY --only-show-errors
