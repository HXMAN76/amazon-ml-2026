SHELL := /bin/bash
BUCKET ?= $(AMLC_BUCKET)

.PHONY: setup test lint mlflow push-code pull-data push-images push-artifacts pull-artifacts

setup:            ## local env (GPU torch + HF + tracking)
	uv sync --all-extras

test:
	uv run pytest -q

lint:
	uv run ruff check src tests

mlflow:           ## local experiment UI at http://127.0.0.1:5000
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db

push-code:        ## snapshot of HEAD for EC2 boxes (aws/40_launch_gpu.sh pulls it)
	git archive --format=tar.gz HEAD | aws s3 cp - s3://$(BUCKET)/code/amlc.tar.gz

pull-data:        ## raw dataset from hub
	aws s3 sync s3://$(BUCKET)/00-raw/ data/raw/ --only-show-errors

push-images:      ## downloaded images + manifest -> hub
	aws s3 sync data/images/ s3://$(BUCKET)/01-images/ --only-show-errors

pull-images:
	aws s3 sync s3://$(BUCKET)/01-images/ data/images/ --only-show-errors

push-artifacts:   ## features, preds, runs log -> hub
	aws s3 sync artifacts/ s3://$(BUCKET)/03-features/artifacts/ --only-show-errors --exclude "runs.jsonl"
	aws s3 cp artifacts/runs.jsonl s3://$(BUCKET)/07-experiments/runs-$${AMLC_MEMBER:-$$(hostname)}.jsonl

pull-artifacts:
	aws s3 sync s3://$(BUCKET)/03-features/artifacts/ artifacts/ --only-show-errors
