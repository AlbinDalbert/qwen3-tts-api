# syntax=docker/dockerfile:1.7
ARG BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HF_HOME=/models/.cache/huggingface \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    QWEN_LOCAL_FILES_ONLY=1 \
    QWEN_MODEL_ID=Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    QWEN_DEVICE=cuda:0 \
    QWEN_DTYPE=bfloat16 \
    QWEN_ATTN_IMPL=sdpa \
    QWEN_DEFAULT_LANGUAGE=Auto \
    QWEN_DEFAULT_SPEAKER=Ryan \
    QWEN_DEFAULT_INSTRUCT=

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    ffmpeg libsndfile1 sox git curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install -r /app/requirements.txt

COPY app /app/app

EXPOSE 8888

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=5 \
  CMD curl -fsS http://127.0.0.1:8888/healthz || exit 1

CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8888"]
