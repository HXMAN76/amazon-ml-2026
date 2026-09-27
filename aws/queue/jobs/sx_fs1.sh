set -ex
source /home/ec2-user/anaconda3/etc/profile.d/conda.sh
export BER_DATA=/home/ec2-user/SageMaker/dataset BER_WORK=/home/ec2-user/SageMaker/work
conda activate ber
aws s3 sync s3://sagemaker-us-east-1-567503593043/ber/code_v8 /home/ec2-user/SageMaker/ber_v8 --delete --exclude 'work/*' --only-show-errors --exact-timestamps
cd /home/ec2-user/SageMaker/ber_v8
export PYTHONPATH=/home/ec2-user/SageMaker/ber_v8/src
R=restore:noise_swap+noise_extra+initials+spelled_legal+glued:0.05
OUT=s3://sagemaker-us-east-1-567503593043/runs
run() { b=$1; n=$2; shift 2; python src/scripts/france_variants.py $b $n --rules "$1" --cap 2>&1 | grep -E "^rule|^typeswap|protect|restore|total|cap 5|kept|wrote|France|Traceback|Error"; python src/scripts/check_submission.py $BER_WORK/output/$n $BER_DATA/test | tail -3; aws s3 cp $BER_WORK/output/$n/matching_results.tsv $OUT/$n/output/matching_results.tsv --only-show-errors; aws s3 cp $BER_WORK/output/$n/candidate_pairs.tsv $OUT/$n/output/candidate_pairs.tsv --only-show-errors; }
run s29 v8w_s29_ARt "typeswap:1.01,typeswap:1.01:0.6:30:300,thrpn:0.995,protect:0.9,$R"
run s29 v8w_s29_ARtL "typeswap:1.01,typeswap:1.01:0.6:30:300,thrpn:0.995,legalx:1.01,protect:0.9,$R"
run s29 v8w_s29_ARtL99 "typeswap:1.01,typeswap:1.01:0.6:30:300,thrpn:0.999,legalx:1.01,protect:0.9,$R"
run s29 v8w_s29_ARtL9 "typeswap:1.01,typeswap:1.01:0.6:30:300,thrpn:0.9999,legalx:1.01,protect:0.9,$R"
