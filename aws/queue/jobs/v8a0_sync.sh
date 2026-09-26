# v8 step a0 (GPU lane jobs, first): the team's exported work directory, copied server-side into our bucket (ber/team_work: the notebook role
# cannot read the team's bucket), -> $SM/work_t; our v6 work/ stays untouched. The marker lets the other jobs start.
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
df -h $SM | tail -1
aws s3 sync s3://$B/ber/team_work $BER_WORK --only-show-errors
du -sh $BER_WORK; df -h $SM | tail -1
ls $BER_WORK $BER_WORK/output $BER_WORK/models $BER_WORK/official
touch $BER_WORK/.v8a_synced
