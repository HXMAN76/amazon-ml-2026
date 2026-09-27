# v8 (CPU lane jobs2): namesakes at another address. The portal-best recipe plus dropping equal-name pairs whose address differs (similarity < 60 / < 80,
# or another house number) below p 0.995 / 0.9999 / any p, each with its slot-fit decoy share; the same categories for the US as the reference of the
# fit's bias (country us).
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
R=restore:noise_swap+noise_extra+initials+spelled_legal+glued
for spec in exactfar:0.995:60 exactfar:0.995:80 exactfar:0.9999:60 exactfar:1.01:60; do
  n=v8x_s27_$(echo $spec | tr ':.' '__')
  python src/scripts/france_variants.py s27 $n --rules typeswap:1.01,thrpn:0.995,$spec,protect:0.9,$R:0.05 --cap 2>&1 | grep -E "exactfar.*fires on|^protect|with matches|^Traceback"
done
python src/scripts/france_variants.py s27 tmp_exactfar_us --country us --rules exactfar:0.995:60,exactfar:0.9999:60,exactfar:1.01:60 2>&1 | grep -E "fires on|^Traceback"
