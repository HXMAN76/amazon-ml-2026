# v8 step r3 (CPU lane jobs2, protect fixed for thrpn): the recipe with France's noise-word swaps treated as true copies: typeswap + thrpn 0.995 (spares equal names,
# initials and noise-word swaps) + protect, then the restore of unowned noise-word swaps, initials, spelled legal forms and glued names at the S1's
# address (exact names are not restored: on the holdout such restores are 0.4% true). On s27 and on s22F12n.
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
R=restore:noise_swap+initials+spelled_legal+glued
for b in s27 s22F12n; do
  python src/scripts/france_variants.py $b v8s_${b}_A --rules typeswap:1.01,thrpn:0.995,protect:0.9 --cap 2>&1 | grep -E "fires on|^protect|^restore|with matches|PASS|FAIL|Traceback"
  python src/scripts/france_variants.py $b v8s_${b}_AR --rules typeswap:1.01,thrpn:0.995,protect:0.9,$R:0.05 --cap 2>&1 | grep -E "fires on|^protect|^restore|with matches|PASS|FAIL|Traceback"
  python src/scripts/france_variants.py $b v8s_${b}_AR2 --rules typeswap:1.01,thrpn:0.995,protect:0.9,$R:0.2 --cap 2>&1 | grep -E "fires on|^protect|^restore|with matches|PASS|FAIL|Traceback"
  for v in A AR; do python src/scripts/france_empty.py $b v8s_${b}_$v 2>&1 | grep -E "^│ france ┆ 259452"; done
done
