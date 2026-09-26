# v8 (CPU lane jobs2): stop cross-encoder 4 (seed averaging adds little now that the France cross-encoder has plateaued) so the GPU goes to the
# Qwen3-0.6B scoring for the US/India stack sooner. Kills the training process first, then its job script.
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
pkill -f "dir xencFR4" || true
sleep 3
pkill -f "jobs-v8m4_xfr4.sh" || true
sleep 3
ps aux | grep -E "xencFR4|v8m4" | grep -v grep | wc -l
nvidia-smi --query-gpu=memory.used --format=csv
