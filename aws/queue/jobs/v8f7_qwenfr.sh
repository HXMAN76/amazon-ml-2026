# v8 (CPU lane jobs2): does the Qwen3-0.6B cross-encoder (xs4, xenc3Q_v7) see France's decoys? Control: its share below 0.5 on typeswap pairs
# (known 84-92% decoys); then France pairs left after typeswap + thrpn 0.995 with xs4 below 0.05 / 0.2 / 0.5: counts, slot fit, examples; US reference.
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
echo "== france control"
python src/scripts/france_variants.py s28 v8_q --dry --rules xfr:xenc3Q_v7:0.5,typeswap:1.01,thrpn:0.995 2>&1 | grep -E "^rule|^xfr|Traceback|Error"
for c in france us; do
  echo "== $c"
  python src/scripts/france_variants.py s28 v8_q --dry --country $c --samples 30 --rules typeswap:1.01,thrpn:0.995,xfr:xenc3Q_v7:0.05,xfr:xenc3Q_v7:0.2,xfr:xenc3Q_v7:0.5 2>&1 | grep -v "^+" | grep -vE "^export|^PS1"
done
