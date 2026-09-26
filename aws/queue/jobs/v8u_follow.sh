# v8 step u (CPU lane jobs2): follow-ups for the portal answers: stricter protected thresholds (thrp 0.99, 0.995) with protect on s22 and on the
# cross-encoder base s22F12n, each with the slot-fit decoy share of what the threshold drops.
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
for b in s22 s22F12n; do for t in 0.99 0.995; do
  python src/scripts/france_variants.py $b v8_${b}_tpp${t#0.} --rules typeswap:1.01,thrp:$t,protect:0.9 --cap 2>&1 | grep -E "fires on|^protect|with matches|PASS|FAIL|Traceback"
done; done
