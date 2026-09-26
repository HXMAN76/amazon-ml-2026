# v8 step q (CPU lane jobs2): candidate recipes on the two finalist bases (s27: best holdout; s22F12n: France cross-encoders 1+2 in the stack):
# typeswap + protected threshold 0.995 + protect, with and without the legal-form sibling rule and the restore of true-copy patterns at the S1's
# address (france_recall.py). Each prints what every rule drops or adds and its slot-fit decoy share.
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
R=restore:exact+spelled_legal+initials+glued+noise_swap
for b in s27 s22F12n; do
  python src/scripts/france_variants.py $b v8q_${b}_T --rules typeswap:1.01,thrp:0.995,protect:0.9 --cap 2>&1 | grep -E "fires on|^protect|^restore|with matches|PASS|FAIL|Traceback"
  python src/scripts/france_variants.py $b v8q_${b}_TL --rules typeswap:1.01,thrp:0.995,legalhouse:1.01,protect:0.9 --cap 2>&1 | grep -E "fires on|^protect|^restore|with matches|PASS|FAIL|Traceback"
  python src/scripts/france_variants.py $b v8q_${b}_TR --rules typeswap:1.01,thrp:0.995,protect:0.9,$R:0.3 --cap 2>&1 | grep -E "fires on|^protect|^restore|with matches|PASS|FAIL|Traceback"
  python src/scripts/france_variants.py $b v8q_${b}_TLR --rules typeswap:1.01,thrp:0.995,legalhouse:1.01,protect:0.9,$R:0.3 --cap 2>&1 | grep -E "fires on|^protect|^restore|with matches|PASS|FAIL|Traceback"
  python src/scripts/france_variants.py $b v8q_${b}_TLR05 --rules typeswap:1.01,thrp:0.995,legalhouse:1.01,protect:0.9,$R:0.05 --cap 2>&1 | grep -E "fires on|^protect|^restore|with matches|PASS|FAIL|Traceback"
done
for n in v8q_s27_T v8q_s27_TLR v8q_s22F12n_TLR; do python src/scripts/france_empty.py s22 $n 2>&1 | grep -E "^│ france ┆ 259452"; done
