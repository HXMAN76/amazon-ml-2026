# v8 delivery 2 (CPU lane jobs2): the noise-aware final recipes: the official validator WITH --check-ids on each chosen file, then both TSVs of the files that pass to
# s3://$B/ber/v8/runs/<name>/output/ (the laptop copies them server-side into the team's handoff folder). FILES = space-separated names.
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
FILES="${FILES:-v8u_s27_AR v8u_s27_A v8u_s22F12n_AR v8u_s22F12n_A}"
for n in $FILES; do
  o=$BER_WORK/output/$n
  if python $BER_WORK/official/validate_submission.py --matching $o/matching_results.tsv --candidate $o/candidate_pairs.tsv --test-dir $BER_DATA/test --check-ids | tail -3 | grep -q "^PASS"; then
    echo "$n: PASS with --check-ids"
    aws s3 cp $o/matching_results.tsv s3://$B/ber/v8/runs/$n/output/matching_results.tsv --only-show-errors
    aws s3 cp $o/candidate_pairs.tsv s3://$B/ber/v8/runs/$n/output/candidate_pairs.tsv --only-show-errors
  else
    echo "$n: VALIDATOR FAILED"; python $BER_WORK/official/validate_submission.py --matching $o/matching_results.tsv --candidate $o/candidate_pairs.tsv --test-dir $BER_DATA/test --check-ids | tail -15
  fi
done
