#!/usr/bin/env bash
# Cost panic button for the CURRENT account. Stops (does not terminate/delete) running EC2
# and SageMaker notebooks, stops running SageMaker jobs, and LISTS endpoints.
#   bash aws/99_stop_all.sh               # stop what is running
#   DELETE_ENDPOINTS=1 bash aws/99_stop_all.sh   # also delete endpoints (irreversible)
set -uo pipefail
cd "$(dirname "$0")" && source ./env.sh
echo "account $(me) region $AWS_REGION"

ids=$(aws ec2 describe-instances --filters Name=instance-state-name,Values=running,pending \
      --query 'Reservations[].Instances[].InstanceId' --output text)
[[ -n "$ids" ]] && { echo "stopping EC2: $ids"; aws ec2 stop-instances --instance-ids $ids >/dev/null; }

for nb in $(aws sagemaker list-notebook-instances --status-equals InService \
            --query 'NotebookInstances[].NotebookInstanceName' --output text); do
  echo "stopping notebook $nb"; aws sagemaker stop-notebook-instance --notebook-instance-name "$nb"
done
for j in $(aws sagemaker list-training-jobs --status-equals InProgress --query 'TrainingJobSummaries[].TrainingJobName' --output text); do
  echo "stopping training job $j"; aws sagemaker stop-training-job --training-job-name "$j"
done
for j in $(aws sagemaker list-processing-jobs --status-equals InProgress --query 'ProcessingJobSummaries[].ProcessingJobName' --output text); do
  echo "stopping processing job $j"; aws sagemaker stop-processing-job --processing-job-name "$j"
done
for j in $(aws sagemaker list-transform-jobs --status-equals InProgress --query 'TransformJobSummaries[].TransformJobName' --output text); do
  echo "stopping transform job $j"; aws sagemaker stop-transform-job --transform-job-name "$j"
done
for ep in $(aws sagemaker list-endpoints --query 'Endpoints[].EndpointName' --output text); do
  if [[ "${DELETE_ENDPOINTS:-0}" == 1 ]]; then
    echo "deleting endpoint $ep"; aws sagemaker delete-endpoint --endpoint-name "$ep"
  else
    echo "ENDPOINT STILL BILLING: $ep  (rerun with DELETE_ENDPOINTS=1)"
  fi
done
echo "note: stopped EC2 still bills EBS storage; terminate when done with it."
