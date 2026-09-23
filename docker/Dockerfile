# Same env locally (podman/docker + NVIDIA toolkit) and on EC2/SageMaker.
# Build only if you actually need containers; EC2 DLAMI + `uv sync` is usually faster.
#   podman build -f docker/Dockerfile -t amlc:latest .
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 HF_HOME=/opt/hf
RUN apt-get update && apt-get install -y --no-install-recommends git curl libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/amlc
COPY pyproject.toml README.md ./
COPY src ./src
# keep the image's CUDA torch; add the rest
RUN pip install -e ".[hf,track]" awscli

# SageMaker training/processing run `python -m ...` via the job's ContainerEntrypoint
ENTRYPOINT []
