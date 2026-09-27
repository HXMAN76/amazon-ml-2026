# v8 (GPU lane): s28L = s28's stack settings + the language-free features; paired holdout test against s28; France recipe of v8u_s28_FIN on s28L
# (lists rebuilt on s28L's own probabilities), validation --check-ids and upload of v8u_s28L_FIN.
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
ls $BER_WORK/stack_eqL/test/chunk_*.parquet >/dev/null 2>&1 || { echo "no stack_eqL chunks"; exit 1; }
python -m ber.stages.stack train --name s28L --base v7 --tag _eqL --set max_depth=9,eta=0.05,rounds=1500,early_stop=50 2>&1 | grep -iE "holdout|threshold|delta|paired|error|Traceback" | tail -8
python -m ber.stages.stack predict --name s28L 2>&1 | grep -E "kept|wrote|PASS|FAIL|Traceback" | tail -4
python src/scripts/paired_models.py s28 s28L 2>&1 | tail -8
python src/scripts/france_variants.py s28L v8u_s28L_ALL2 --rules "typeswap:1.01,typeswap:1.01:0.6:30:300,thrpn:0.9999,legalx:1.01,protect:0.9,restore:noise_swap+noise_extra+initials+spelled_legal+glued:0.05" --cap 2>&1 | grep -E "^rule|^alias|total dropped|wrote|Traceback" | cut -c1-200
python src/scripts/france_lists.py v8u_s28L_ALL2 v8x/L s28L 2>&1 | grep -E "^fb_|Traceback"
python src/scripts/france_variants.py s28L v8u_s28L_FIN --rules "typeswap:1.01,typeswap:1.01:0.6:30:300,thrpn:0.9999,legalx:1.01,droplist:L/fb_ns_ref,droplist:L/fb_nsnear_ref,protect:0.9,restore:noise_swap+noise_extra+initials+spelled_legal+glued:0.05,addlist:L/fb_coined_hi" --cap 2>&1 | grep -E "^rule|^alias|^addlist|total dropped|wrote|Traceback" | cut -c1-200
n=v8u_s28L_FIN; o=$BER_WORK/output/$n
python $BER_WORK/official/validate_submission.py --matching $o/matching_results.tsv --candidate $o/candidate_pairs.tsv --test-dir $BER_DATA/test --check-ids | tail -3 | grep -q "^PASS" \
  && { echo "$n: PASS with --check-ids"; aws s3 cp $o/ s3://$B/ber/v8/runs/$n/output/ --recursive --exclude "*" --include "*.tsv" --only-show-errors; } || echo "$n: VALIDATOR FAILED"
