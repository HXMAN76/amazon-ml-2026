# v8: v8u_s28_FIN = ALL2's rules + fork B's refined namesake drops (fb_ns_ref: exact core on another street, name on 11+ France S1 and p < 0.9999 or
# 6+ and p < 0.99; fb_nsnear_ref) + re-adding the coined/glued copies in [0.995, 0.9999) the 0.9999 cut drops (fb_coined_hi); no coined rule.
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
n=v8u_s28_FIN
python src/scripts/france_variants.py s28 $n --rules "typeswap:1.01,typeswap:1.01:0.6:30:300,thrpn:0.9999,legalx:1.01,droplist:fb_ns_ref,droplist:fb_nsnear_ref,protect:0.9,restore:noise_swap+noise_extra+initials+spelled_legal+glued:0.05,addlist:fb_coined_hi" --cap 2>&1 | grep -E "^rule|^alias|protect|restore .* adds|^addlist|total dropped|cap 5|wrote|Traceback|Error" | cut -c1-230
o=$BER_WORK/output/$n
python $BER_WORK/official/validate_submission.py --matching $o/matching_results.tsv --candidate $o/candidate_pairs.tsv --test-dir $BER_DATA/test --check-ids | tail -3 | grep -q "^PASS" \
  && { echo "$n: PASS with --check-ids"; aws s3 cp $o/ s3://$B/ber/v8/runs/$n/output/ --recursive --exclude "*" --include "*.tsv" --only-show-errors; } || echo "$n: VALIDATOR FAILED"
