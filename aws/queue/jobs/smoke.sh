# Smoke job: GPU, env and the test suite on the notebook (a few minutes).
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
python -c "import torch, xgboost; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'xgboost', xgboost.__version__)"
nproc; free -g | head -2; df -h $SM | tail -1
python -m pytest -q -p no:warnings src/tests
