# v8 (CPU lane jobs2): v8u_s28_AR with the France protected cut-off raised from 0.995 to 0.9999 (v8u_s28_AR9) and to 0.999 (v8u_s28_AR99), same
# typeswap / protect / restore / cap; portal test of France's over-confidence above 0.995 (slot fit 0.2 to 0.55 there, not trusted alone).
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
R="protect:0.9,restore:noise_swap+noise_extra+initials+spelled_legal+glued:0.05"
python src/scripts/france_variants.py s28 v8u_s28_AR9 --rules "typeswap:1.01,thrpn:0.9999,$R" --cap 2>&1 | grep -E "^rule|protect|restore|total dropped|cap 5|kept|wrote|Traceback|Error"
python src/scripts/france_variants.py s28 v8u_s28_AR99 --rules "typeswap:1.01,thrpn:0.999,$R" --cap 2>&1 | grep -E "^rule|protect|restore|total dropped|cap 5|kept|wrote|Traceback|Error"
for n in v8u_s28_AR9 v8u_s28_AR99; do
  o=$BER_WORK/output/$n
  python $BER_WORK/official/validate_submission.py --matching $o/matching_results.tsv --candidate $o/candidate_pairs.tsv --test-dir $BER_DATA/test --check-ids | tail -3 | grep -q "^PASS" \
    && { echo "$n: PASS with --check-ids"; aws s3 cp $o/ s3://$B/ber/v8/runs/$n/output/ --recursive --exclude "*" --include "*.tsv" --only-show-errors; } || echo "$n: VALIDATOR FAILED"
done
