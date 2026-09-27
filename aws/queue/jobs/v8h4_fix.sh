# v8: legalx on French legal forms only + alias protection (pool alias part holding the S1 core): v8u_s28_ARtL2 / ALL2, then nsaway (N files),
# and what coined would still restore on top of ARtL2 (by p band). Validation --check-ids and upload of the N files.
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
python src/scripts/france_variants.py s28 v8u_s28_ARtL2 --rules "typeswap:1.01,typeswap:1.01:0.6:30:300,thrpn:0.995,legalx:1.01,protect:0.9,restore:noise_swap+noise_extra+initials+spelled_legal+glued:0.05" --cap 2>&1 | grep -E "^rule|^alias|protect|restore .* adds|total dropped|wrote|Traceback|Error" | cut -c1-220
python src/scripts/france_variants.py s28 v8u_s28_ALL2 --rules "typeswap:1.01,typeswap:1.01:0.6:30:300,thrpn:0.9999,legalx:1.01,protect:0.9,restore:noise_swap+noise_extra+initials+spelled_legal+glued:0.05" --cap 2>&1 | grep -E "^rule legalx|^alias|total dropped|wrote|Traceback|Error" | cut -c1-220
python src/scripts/france_post.py s28 v8u_s28_ARtL2 v8u_s28_ARtL2N --rules nsaway 2>&1 | grep -E "^rule|wrote|Traceback"
python src/scripts/france_post.py s28 v8u_s28_ALL2 v8u_s28_ALL2N --rules nsaway 2>&1 | grep -E "^rule|wrote|Traceback"
python src/scripts/france_post.py s28 v8u_s28_ARtL2 v8u_s28_ARtL2C --rules coined 2>&1 | grep -E "^rule|p [01]\.|wrote|Traceback" | head -30
for n in v8u_s28_ARtL2N v8u_s28_ALL2N; do
  o=$BER_WORK/output/$n
  python $BER_WORK/official/validate_submission.py --matching $o/matching_results.tsv --candidate $o/candidate_pairs.tsv --test-dir $BER_DATA/test --check-ids | tail -3 | grep -q "^PASS" \
    && { echo "$n: PASS with --check-ids"; aws s3 cp $o/ s3://$B/ber/v8/runs/$n/output/ --recursive --exclude "*" --include "*.tsv" --only-show-errors; } || echo "$n: VALIDATOR FAILED"
done
