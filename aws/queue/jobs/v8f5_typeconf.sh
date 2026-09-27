# v8 (CPU lane jobs2): type-word conflicts beyond single swaps (rule typeconf) on s28's France pairs after typeswap and the 0.995 cut: counts,
# slot fit and raw examples, with the learned type-word list; the US as reference.
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
for c in france us; do
  echo "== $c"
  python src/scripts/france_variants.py s28 v8_tc --dry --country $c --samples 30 --rules typeswap:1.01,thrpn:0.995,typeconf:1.01,typeconf:1.01:0.5,typeconf:1.01:0.3 2>&1 | grep -v "^+"
done
