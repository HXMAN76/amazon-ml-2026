# v8 (CPU lane jobs2): final recipe candidates: typeswap + thrpn 0.995 (spares equal names, initials, noise-word swaps and additions) + protect,
# plus the restore of unowned noise-word swaps and additions, initials, spelled legal forms and glued names at the S1's address. On s27 and s22F12n.
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
R=restore:noise_swap+noise_extra+initials+spelled_legal+glued
for b in s27 s22F12n; do
  python src/scripts/france_variants.py $b v8u_${b}_A --rules typeswap:1.01,thrpn:0.995,protect:0.9 --cap 2>&1 | grep -E "fires on|^protect|^restore|with matches|PASS|FAIL|^Traceback"
  python src/scripts/france_variants.py $b v8u_${b}_AR --rules typeswap:1.01,thrpn:0.995,protect:0.9,$R:0.05 --cap 2>&1 | grep -E "fires on|^protect|^restore|with matches|PASS|FAIL|^Traceback"
  python src/scripts/france_empty.py $b v8u_${b}_AR 2>&1 | grep -E "^│ france ┆ 259452"
done
