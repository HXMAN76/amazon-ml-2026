# v8 (GPU lane jobs, right after the Qwen scoring): s28 = s27's stack recipe (symmetric cross-encoder seeds as xs/xs_asym/xs_seed_gap, e5-small band score xs2, e5-base
# score xs3) plus the Qwen3-0.6B score as xs4, trained on the same 1.5M S1 (s24, the same test on s22, is skipped to save time: the paired test against s27 decides).
# Paired tests against s27, the portal-best France recipe on s28, validation --check-ids and upload.
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
aws s3 sync s3://$B/ber/team_work/xenc2SymE_v7 $BER_WORK/xenc2SymE_v7 --only-show-errors
A="--base v7 --tag _eq --xenc --xcons --xenc-fit-more 300000 --decoy --extra --sub-q 1500000 --xenc-dir xenc2SymE_v7 --xenc-dir2 xenc_v7 --xenc-dir3 xenc2F_v7 --xenc-dir4 xenc3Q_v7"
python -m ber.stages.stack build --split train $A
python -m ber.stages.stack tfidf --split train --tag _eq
python -m ber.stages.stack build --split test $A
python -m ber.stages.stack tfidf --split test --tag _eq
python -m ber.stages.stack train --name s28 --base v7 --tag _eq --set max_depth=9,eta=0.05,rounds=1500,early_stop=50
python -m ber.stages.stack predict --name s28
python src/scripts/paired_models.py s27 s28
R=restore:noise_swap+noise_extra+initials+spelled_legal+glued
python src/scripts/france_variants.py s28 v8u_s28_AR --rules typeswap:1.01,thrpn:0.995,protect:0.9,$R:0.05 --cap 2>&1 | grep -E "fires on|^protect|^restore adds|with matches|PASS|FAIL|^Traceback"
o=$BER_WORK/output/v8u_s28_AR
if python $BER_WORK/official/validate_submission.py --matching $o/matching_results.tsv --candidate $o/candidate_pairs.tsv --test-dir $BER_DATA/test --check-ids | tail -3 | grep -q "^PASS"; then
  echo "v8u_s28_AR: PASS with --check-ids"; aws s3 cp $o/ s3://$B/ber/v8/runs/v8u_s28_AR/output/ --recursive --exclude "*" --include "*.tsv" --only-show-errors
else echo "v8u_s28_AR: VALIDATOR FAILED"; fi
