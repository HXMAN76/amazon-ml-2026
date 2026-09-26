# v8 step r (CPU lane jobs2): France's missing true copies. Shortlisted pairs below the decision whose pool record nobody owns, by name relation
# (exact, spelled legal, initials, glued, noise-word swap) and address evidence, with the France cross-encoder score; the same restore
# candidates on the labelled holdout give their true share.
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
aws s3 sync s3://$B/ber/team_work/features/train_rest $BER_WORK/features/train_rest --only-show-errors
python src/scripts/france_recall.py s22 xencF12
