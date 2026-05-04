FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev \
    git wget curl \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python

RUN pip install --no-cache-dir \
    torch==2.4.0 \
    torchvision==0.19.0 \
    --extra-index-url https://download.pytorch.org/whl/cu124

RUN pip install --no-cache-dir \
    runpod \
    safetensors \
    Pillow \
    einops \
    transformers \
    huggingface_hub \
    fire \
    openai

WORKDIR /app
COPY . .

ENV PYTHONPATH=/app/src

CMD ["python", "handler.py"]
