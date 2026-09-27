# v8 (CPU lane jobs2): rule legal (legal_conflict feature: both names carry a legal form and they differ) after the current France recipe's rules:
# extra France pairs, slot fit, examples; US and India reference.
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
for c in france us india; do
  echo "== $c"
  python src/scripts/france_variants.py s28 v8_lg --dry --country $c --samples 25 --rules typeswap:1.01,typeswap:1.01:0.6:30:300,thrpn:0.995,legal:1.01 2>&1 | grep -v "^+" | grep -vE "^export|^PS1|^┌|^└|^╞|^├|shape:|^│ ---|^│ xb|^│ str" | grep -A 30 "rule legal"
done
