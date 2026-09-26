# v8 (GPU lane jobs): stop the s24 job that waits on the CPU lane for the Qwen scores (it blocks every CPU job behind it); the s24 stack now runs on
# this lane right after the Qwen scoring (v8m3r_s24).
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
pkill -f "jobs2-v8z9_s24.sh" || true
sleep 5
ps aux | grep -c "jobs2-v8z9_s24.sh" || true
