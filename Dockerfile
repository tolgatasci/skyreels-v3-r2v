FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/root/.cache/huggingface

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    git wget curl ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Upgrade PyTorch to 2.8+ (SkyReels-V3 requirement)
# NOTE: If cu121 build not available, change to:
#   --index-url https://download.pytorch.org/whl/cu124
RUN pip install --no-cache-dir \
    torch==2.8.0 \
    torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Build flash_attn from source (matches torch + CUDA)
RUN pip install --no-cache-dir flash-attn --no-build-isolation

# SkyReels-V3 core dependencies
RUN pip install --no-cache-dir \
    diffusers==0.34.0 \
    transformers==4.53.2 \
    accelerate \
    tokenizers==0.21.4 \
    safetensors \
    sentencepiece \
    protobuf \
    Pillow \
    runpod \
    huggingface_hub \
    imageio \
    imageio-ffmpeg \
    "numpy==1.26.4" \
    einops \
    easydict \
    ftfy \
    tqdm \
    opencv-python-headless \
    kornia \
    omegaconf \
    torchao==0.10.0 \
    pyloudnorm \
    librosa \
    moviepy==2.2.1

# Clone SkyReels-V3 repository
RUN git clone https://github.com/SkyworkAI/SkyReels-V3.git /app/SkyReels-V3

COPY handler.py /app/handler.py

CMD ["python3", "-u", "handler.py"]
