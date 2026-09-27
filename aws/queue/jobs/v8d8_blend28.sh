# v8 (CPU lane jobs2): logit-mean blends s28+s28b and s28+s27, paired holdout tests against s28; the France recipe on a blend that wins.
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
R=restore:noise_swap+noise_extra+initials+spelled_legal+glued
for spec in "s28m s28 s28b" "s28n s28 s27"; do set -- $spec
  python src/scripts/blend_stacks.py $1 $2 $3 2>&1 | tail -3 || continue
  python src/scripts/paired_models.py s28 $1 | tee /tmp/p_$1.json | grep -E "delta|B_f05" -A2
  if python -c "import json,sys; d=json.load(open('/tmp/p_$1.json')); sys.exit(0 if d['delta_ci95'][0] > 0 else 1)"; then
    python src/scripts/france_variants.py $1 v8u_${1}_AR --rules typeswap:1.01,thrpn:0.995,protect:0.9,$R:0.05 --cap 2>&1 | grep -E "fires on|with matches|PASS|FAIL|^Traceback"
    o=$BER_WORK/output/v8u_${1}_AR
    python $BER_WORK/official/validate_submission.py --matching $o/matching_results.tsv --candidate $o/candidate_pairs.tsv --test-dir $BER_DATA/test --check-ids | tail -3 | grep -q "^PASS" \
      && { echo "v8u_${1}_AR: PASS with --check-ids"; aws s3 cp $o/ s3://$B/ber/v8/runs/v8u_${1}_AR/output/ --recursive --exclude "*" --include "*.tsv" --only-show-errors; } || echo "v8u_${1}_AR: VALIDATOR FAILED"
  else echo "$1 does not beat s28"; fi
done
