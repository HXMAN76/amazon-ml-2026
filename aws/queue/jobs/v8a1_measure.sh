# v8 step a1 (CPU lane jobs2): reproduction of s22t2c (must be identical), then label-free France measurements on s22: S1 emptied by the rules,
# relation types, decoy share by probability zone; paired holdout tests of the overnight bases s26 and s22e against s22.
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
while [ ! -f $BER_WORK/.v8a_synced ]; do sleep 30; done
python src/scripts/france_variants.py s22 v8_t2c_re --rules typeswap:1.01,thr:0.985 --cap
if cmp -s <(sort $BER_WORK/output/v8_t2c_re/matching_results.tsv) <(sort $BER_WORK/output/s22t2c/matching_results.tsv); then echo "REPRO s22t2c: IDENTICAL"; else
  echo "REPRO s22t2c: DIFFERENT"; diff <(sort $BER_WORK/output/v8_t2c_re/matching_results.tsv) <(sort $BER_WORK/output/s22t2c/matching_results.tsv) | wc -l; fi
python src/scripts/france_empty.py s22 s22t2c
python src/scripts/paired_models.py s22 s26
python src/scripts/paired_models.py s22 s22e
python src/scripts/relations.py s22
python src/scripts/decoy_by_pzone.py s22
