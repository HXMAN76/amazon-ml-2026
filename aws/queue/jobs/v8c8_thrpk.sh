# v8 (CPU lane jobs2): the protected cut-off thrpk (also spares fuzzy-glued names, reordered words, words dropped, noise or `france` added, 4-letter
# initials, several words changed with some shared; `alias`: coined aliases of an S1 alone at its address), on s27 and s22F12n, with protect and the
# restore of France copy kinds; validation with --check-ids and upload of the files.
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
R=restore:noise_swap+noise_extra+initials+spelled_legal+glued
for b in s27 s22F12n; do
  python src/scripts/france_variants.py $b v8k_${b}_K --rules typeswap:1.01,thrpk:0.995,protect:0.9,$R:0.05 --cap 2>&1 | grep -E "fires on|^protect|^restore adds|with matches|PASS|FAIL|^Traceback"
  python src/scripts/france_variants.py $b v8k_${b}_KA --rules typeswap:1.01,thrpk:0.995:alias,protect:0.9,$R:0.05 --cap 2>&1 | grep -E "fires on|^protect|^restore adds|with matches|PASS|FAIL|^Traceback"
  python src/scripts/france_empty.py $b v8k_${b}_KA 2>&1 | grep -E "^│ france ┆ 259452"
done
for n in v8k_s27_K v8k_s27_KA v8k_s22F12n_K v8k_s22F12n_KA; do
  o=$BER_WORK/output/$n
  [ -f $o/matching_results.tsv ] || continue
  if python $BER_WORK/official/validate_submission.py --matching $o/matching_results.tsv --candidate $o/candidate_pairs.tsv --test-dir $BER_DATA/test --check-ids | tail -3 | grep -q "^PASS"; then
    echo "$n: PASS with --check-ids"; aws s3 cp $o/ s3://$B/ber/v8/runs/$n/output/ --recursive --exclude "*" --include "*.tsv" --only-show-errors
  else echo "$n: VALIDATOR FAILED"; fi
done
