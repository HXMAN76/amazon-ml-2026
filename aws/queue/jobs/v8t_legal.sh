# v8 step t (CPU lane jobs2): siblings next door. Pairs with a legal-form conflict and another house number: their share in France, the US and the
# labelled holdout, their holdout true share and the implied France wrong share; then the best recipes with the legalhouse rule added.
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
aws s3 sync s3://$B/ber/team_work/features/train_rest $BER_WORK/features/train_rest --only-show-errors
python src/scripts/legal_house.py s22
python src/scripts/france_variants.py s22 v8_tppl --rules typeswap:1.01,thrp:0.985,legalhouse:1.01,protect:0.9 --cap 2>&1 | grep -E "fires on|^protect|with matches|PASS|FAIL|Traceback"
python src/scripts/france_variants.py s27 v8_s27_tppl --rules typeswap:1.01,thrp:0.985,legalhouse:1.01,protect:0.9 --cap 2>&1 | grep -E "fires on|^protect|with matches|PASS|FAIL|Traceback"
