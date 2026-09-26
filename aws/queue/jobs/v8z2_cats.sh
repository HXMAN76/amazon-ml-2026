# v8 (GPU lane, CPU work while the GPU is idle): slot-fit decoy shares of candidate decoy categories in France against the US (the fit is biased,
# so only a France share well above the US share means decoys): legal-form conflict, the same plus another house number, weak names.
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
for c in france us; do
  python src/scripts/france_variants.py s22 tmp_cats_$c --country $c --rules legal:1.01,legalhouse:1.01,weak:80:1.01,weak:60:1.01,thrx:0.985,thrp:0.985 2>&1 | grep -E "^rule|^total"
done
