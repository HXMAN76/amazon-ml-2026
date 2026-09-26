# v8 step t2 (CPU lane jobs2): blends of the team's stacks (mean logit; same shortlist): s22+s27 and s22+s26+s27+s22e, paired holdout tests against
# s22 and s27 (labelled: US and India, 85% of the score); the France recipe on a blend that wins.
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
python src/scripts/blend_stacks.py sB2 s22 s27 2>&1 | grep -vE "^\s*$" | tail -5
python src/scripts/blend_stacks.py sB4 s22 s26 s27 s22e 2>&1 | grep -vE "^\s*$" | tail -5
for m in sB2 sB4; do python src/scripts/paired_models.py s27 $m | grep -E "delta|ci95" -A2; done
for m in sB2 sB4; do python src/scripts/france_variants.py $m v8_${m}_tpp --rules typeswap:1.01,thrp:0.985,protect:0.9 --cap 2>&1 | grep -E "fires on|^protect|with matches|PASS|FAIL|Traceback"; done
