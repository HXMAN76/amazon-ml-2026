# v8 (CPU lane jobs2): typeswap with rarer learned type words (slot ratio >= 0.6 over >= 30 pairs, both words naming >= 300 France S1, noise words and
# abbreviations out) after the original typeswap and the 0.995 cut: extra pairs, slot fit, examples; US reference for the same rule.
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
for c in france us; do
  echo "== $c"
  python src/scripts/france_variants.py s28 v8_ts2 --dry --country $c --samples 40 --rules typeswap:1.01,thrpn:0.995,typeswap:1.01:0.6:30:300,typeswap:1.01:0.5:20:300 2>&1 | grep -v "^+" | grep -vE "^export|^PS1|^│ xb|^│ str|^│ ---|^┌|^└|^╞|^├|shape"
done
