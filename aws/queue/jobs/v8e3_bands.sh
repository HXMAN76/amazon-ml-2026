# v8 (CPU lane jobs2): France decoy share (slot fit) of the thrpn-eligible pairs of s28 by probability band, after typeswap: is the cut-off 0.995
# too low or too high? A band above the cut-off with a decoy share over 26% is worth dropping; a band below it under 26% is worth keeping.
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
python src/scripts/france_variants.py s28 v8_bands --dry --rules typeswap:1.01,thrpn:0.9:0.7,thrpn:0.95:0.9,thrpn:0.98:0.95,thrpn:0.99:0.98,thrpn:0.995:0.99,thrpn:0.998:0.995,thrpn:0.999:0.998,thrpn:0.9995:0.999,thrpn:0.9999:0.9995,thrpn:1.01:0.9999 2>&1 | grep -E "^rule|typeswap:|Traceback|Error"
