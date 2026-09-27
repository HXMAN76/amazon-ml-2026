# v8 (GPU lane): language-free sibling / noise features (stack_langfree.py) on s28's stack chunks -> stack_eqL (disk checked first).
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
du -sh $BER_WORK/stack_eq; df -h $BER_WORK | tail -1
need=$(du -sk $BER_WORK/stack_eq | cut -f1); free=$(df -k $BER_WORK | tail -1 | awk '{print $4}')
if [ $free -lt $((need + need / 5)) ]; then echo "NOT ENOUGH DISK: need $need kB, free $free kB"; exit 1; fi
python src/scripts/stack_langfree.py _eq _eqL 2>&1 | grep -vE "Deprecation|empty_as_null" | tail -12
