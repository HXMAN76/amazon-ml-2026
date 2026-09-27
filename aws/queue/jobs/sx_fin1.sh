set -ex
source /home/ec2-user/anaconda3/etc/profile.d/conda.sh
export BER_DATA=/home/ec2-user/SageMaker/dataset BER_WORK=/home/ec2-user/SageMaker/work
conda activate ber
aws s3 sync s3://sagemaker-us-east-1-567503593043/ber/code_v8 /home/ec2-user/SageMaker/ber_v8 --delete --exclude 'work/*' --only-show-errors --exact-timestamps
cd /home/ec2-user/SageMaker/ber_v8
export PYTHONPATH=/home/ec2-user/SageMaker/ber_v8/src
ls $BER_WORK/models/s29/config.json $BER_WORK/output/s29/pair_p.parquet; ls $BER_WORK/features/test | head -3; ls $BER_WORK/features/test | wc -l
mkdir -p $BER_WORK/official
[ -f $BER_WORK/official/validate_submission.py ] || aws s3 cp s3://ml-challenge-nooglers/ml-challenge-2026/raw/v1/utils/validate_submission.py $BER_WORK/official/validate_submission.py --only-show-errors
R="typeswap:1.01,typeswap:1.01:0.6:30:300,thrpn:0.9999,legalx:1.01"
E="protect:0.9,restore:noise_swap+noise_extra+initials+spelled_legal+glued:0.05"
python src/scripts/france_variants.py s29 v8w_s29_ALL2 --rules "$R,$E" --cap 2>&1 | grep -vE "^\s*$|shape:|^[┌└╞├│]"
python src/scripts/france_lists.py v8w_s29_ALL2 v8x/s29 s29
python src/scripts/france_variants.py s29 v8w_s29_FIN --rules "$R,droplist:s29/fb_ns_ref,droplist:s29/fb_nsnear_ref,$E,addlist:s29/fb_coined_hi" --cap 2>&1 | grep -vE "^\s*$|shape:|^[┌└╞├│]"
o=$BER_WORK/output/v8w_s29_FIN
python $BER_WORK/official/validate_submission.py --matching $o/matching_results.tsv --candidate $o/candidate_pairs.tsv --test-dir $BER_DATA/test --check-ids
OUT=s3://sagemaker-us-east-1-567503593043/runs/v8w_s29_FIN/output
aws s3 cp $o/matching_results.tsv $OUT/matching_results.tsv --only-show-errors
aws s3 cp $o/candidate_pairs.tsv $OUT/candidate_pairs.tsv --only-show-errors
