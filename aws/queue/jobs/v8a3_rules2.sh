# v8 step a3 (CPU lane jobs2): more label-free France rule variants on s22, each printing its slot-fit decoy share: the type-word insertion rule,
# higher France cut-offs and the team's protected threshold (thrp), each with the per-S1 protect option.
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
while [ ! -f $BER_WORK/.v8a_synced ]; do sleep 30; done
python src/scripts/france_variants.py s22 v8_ins --rules typeswap:1.01,typeins:1.01,thr:0.985,protect:0.9 --cap
python src/scripts/france_variants.py s22 v8_t990p --rules typeswap:1.01,thr:0.99,protect:0.9 --cap
python src/scripts/france_variants.py s22 v8_t995p --rules typeswap:1.01,thr:0.995,protect:0.9 --cap
python src/scripts/france_variants.py s22 v8_tp985p --rules typeswap:1.01,thrp:0.985,protect:0.9 --cap
python src/scripts/france_variants.py s22 v8_tp995p --rules typeswap:1.01,thrp:0.995,protect:0.9 --cap
