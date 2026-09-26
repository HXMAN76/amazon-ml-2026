# v8 step v (CPU lane jobs2): the protected-threshold recipe (typeswap + thrp 0.985 + protect 0.9 + cap) on the other bases: s27 and the
# France cross-encoder bases, each printing the slot-fit decoy share of what thrp still drops there.
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
for b in s27 s22F12n s22F1n s22F2n; do
  python src/scripts/france_variants.py $b v8_${b}_tpp --rules typeswap:1.01,thrp:0.985,protect:0.9 --cap 2>&1 | grep -E "fires on|^protect|^total|with matches|PASS|FAIL|Traceback"
  python src/scripts/france_empty.py s22 v8_${b}_tpp 2>&1 | grep -E "^│ france ┆ 259452"
done
