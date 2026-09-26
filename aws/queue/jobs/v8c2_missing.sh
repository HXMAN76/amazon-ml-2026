# v8 (CPU lane jobs2): raw examples of France's likeliest missed copies ('other' restore candidates at the S1's address, p >= 0.3), US alongside.
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
python src/scripts/missing_samples.py s27
