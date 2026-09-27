# v8 (GPU lane jobs): s28b = s28's chunks (stack_eq) with a deeper, slower XGBoost (depth 10, eta 0.03, up to 3000 rounds); paired test against s28;
# if it wins, the France recipe, validation --check-ids and upload of v8u_s28b_AR.
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
python -m ber.stages.stack train --name s28b --base v7 --tag _eq --set max_depth=10,eta=0.03,rounds=3000,early_stop=80
python -m ber.stages.stack predict --name s28b
python src/scripts/paired_models.py s28 s28b | tee /tmp/paired.json
python - <<PY || { echo "s28b does not beat s28: stop"; exit 0; }
import json, sys
d = json.load(open("/tmp/paired.json"))
print("delta", d["delta_B_minus_A"], d["delta_ci95"])
sys.exit(0 if d["delta_ci95"][0] > 0 else 1)
PY
R=restore:noise_swap+noise_extra+initials+spelled_legal+glued
python src/scripts/france_variants.py s28b v8u_s28b_AR --rules typeswap:1.01,thrpn:0.995,protect:0.9,$R:0.05 --cap 2>&1 | grep -E "fires on|^protect|^restore adds|with matches|PASS|FAIL|^Traceback"
o=$BER_WORK/output/v8u_s28b_AR
if python $BER_WORK/official/validate_submission.py --matching $o/matching_results.tsv --candidate $o/candidate_pairs.tsv --test-dir $BER_DATA/test --check-ids | tail -3 | grep -q "^PASS"; then
  echo "v8u_s28b_AR: PASS with --check-ids"; aws s3 cp $o/ s3://$B/ber/v8/runs/v8u_s28b_AR/output/ --recursive --exclude "*" --include "*.tsv" --only-show-errors
else echo "v8u_s28b_AR: VALIDATOR FAILED"; fi
