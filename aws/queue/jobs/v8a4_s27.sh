# v8 step a4 (CPU lane jobs2): s27 (two seeds of the symmetric cross-encoder, holdout 0.99063) from the team's second export: paired holdout test
# against s22 and the portal-best France recipe (typeswap + threshold 0.985 + cap), with and without the per-S1 protect option.
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
aws s3 sync s3://$B/ber/team_work/models/s27 $BER_WORK/models/s27 --only-show-errors
aws s3 sync s3://$B/ber/team_work/output/s27 $BER_WORK/output/s27 --only-show-errors
python src/scripts/paired_models.py s22 s27
python src/scripts/france_variants.py s27 v8_27t2c --rules typeswap:1.01,thr:0.985 --cap
python src/scripts/france_variants.py s27 v8_27t2cp --rules typeswap:1.01,thr:0.985,protect:0.9 --cap
python src/scripts/france_empty.py s22 v8_27t2c
