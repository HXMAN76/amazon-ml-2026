# v8 (CPU lane jobs2): rule legalx (both legal-form sets non-empty and disjoint, spaced forms included) after the current France recipe: extra France
# pairs, slot fit, examples; US and India counts for reference (the model already rejects these there).
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
for c in france us india; do
  echo "== $c"
  python src/scripts/france_variants.py s28 v8_lx --dry --country $c --samples 30 --rules typeswap:1.01,typeswap:1.01:0.6:30:300,thrpn:0.995,legalx:1.01 2>&1 | grep -v "^+" | grep -vE "^export|^PS1|^┌|^└|^╞|^├|shape:|^│ ---|^│ xb|^│ str" | grep -A 32 "rule legalx"
done
