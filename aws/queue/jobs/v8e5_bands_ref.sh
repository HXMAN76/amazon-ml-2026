# v8 (CPU lane jobs2): the same bands as v8e3_bands for the US and India (reference: the slot fit should read about 0 where the model is right).
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
for c in us india; do
  echo "== $c"
  python src/scripts/france_variants.py s28 v8_bands --dry --country $c --rules typeswap:1.01,thrpn:0.9:0.7,thrpn:0.95:0.9,thrpn:0.98:0.95,thrpn:0.99:0.98,thrpn:0.995:0.99,thrpn:0.998:0.995,thrpn:0.999:0.998,thrpn:0.9995:0.999,thrpn:0.9999:0.9995,thrpn:1.01:0.9999 2>&1 | grep -E "^rule|typeswap:|Traceback|Error"
done
