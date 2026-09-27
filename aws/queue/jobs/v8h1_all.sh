# v8 (CPU lane jobs2): v8u_s28_ALL = every French precision bet at once (larger type list, legal-form rule, cut 0.9999): the highest-upside file.
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
n=v8u_s28_ALL
python src/scripts/france_variants.py s28 $n --rules "typeswap:1.01,typeswap:1.01:0.6:30:300,thrpn:0.9999,legalx:1.01,protect:0.9,restore:noise_swap+noise_extra+initials+spelled_legal+glued:0.05" --cap 2>&1 | grep -E "^rule|^typeswap|protect|restore|total dropped|cap 5|kept|wrote|Traceback|Error"
o=$BER_WORK/output/$n
python $BER_WORK/official/validate_submission.py --matching $o/matching_results.tsv --candidate $o/candidate_pairs.tsv --test-dir $BER_DATA/test --check-ids | tail -3 | grep -q "^PASS" \
  && { echo "$n: PASS with --check-ids"; aws s3 cp $o/ s3://$B/ber/v8/runs/$n/output/ --recursive --exclude "*" --include "*.tsv" --only-show-errors; } || echo "$n: VALIDATOR FAILED"
