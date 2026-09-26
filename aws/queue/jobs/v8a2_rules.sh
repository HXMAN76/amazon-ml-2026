# v8 step a2 (CPU lane jobs2): the per-S1 protect option on the portal-best recipe (s22 + typeswap + France threshold 0.985 + cap), so its effect
# can be read on the portal against s22t2c alone: an S1 that the threshold would empty keeps its best pair if p >= 0.9 (or 0.95).
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
while [ ! -f $BER_WORK/.v8a_synced ]; do sleep 30; done
python src/scripts/france_variants.py s22 v8_t2cp90 --rules typeswap:1.01,thr:0.985,protect:0.9 --cap
python src/scripts/france_variants.py s22 v8_t2cp95 --rules typeswap:1.01,thr:0.985,protect:0.95 --cap
python src/scripts/france_empty.py s22 v8_t2cp90
python src/scripts/france_variants.py s26 v8_26t2c --rules typeswap:1.01,thr:0.985 --cap
python src/scripts/france_variants.py s22e v8_22et2c --rules typeswap:1.01,thr:0.985 --cap
