FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3.12 python3.12-pip python3.12-dev \
    git wget curl \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.12 /usr/bin/python && \
    ln -sf /usr/bin/python3.12 /usr/bin/python3

RUN pip install --no-cache-dir \
    runpod \
    torch==2.3.0 \
    torchvision \
    safetensors \
    Pillow \
    einops \
    transformers \
    huggingface_hub

WORKDIR /app
COPY . .
RUN pip install -e . --extra-index-url https://download.pytorch.org/whl/cu121

COPY handler.py .

CMD ["python", "handler.py"]
