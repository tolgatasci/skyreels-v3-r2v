FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/root/.cache/huggingface

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    git wget curl ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Upgrade PyTorch to 2.6.0 (latest on cu124, has add_safe_globals + torch.int1)
RUN pip install --no-cache-dir --upgrade \
    torch==2.6.0 \
    torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Skip flash-attn (source build takes 1h+), use PyTorch SDPA instead
ENV ATTN_BACKEND=sdpa

# ALL SkyReels-V3 dependencies (requirements.txt + source code imports)
RUN pip install --no-cache-dir \
    diffusers==0.34.0 \
    transformers==4.53.2 \
    accelerate==1.8.1 \
    tokenizers==0.21.4 \
    torchao==0.10.0 \
    numpy==1.26.4 \
    safetensors \
    sentencepiece \
    protobuf \
    Pillow \
    runpod \
    huggingface_hub \
    imageio \
    imageio-ffmpeg==0.5.1 \
    einops \
    easydict \
    ftfy==6.3.1 \
    tqdm==4.67.1 \
    opencv-python-headless \
    kornia \
    omegaconf==2.3.0 \
    pyloudnorm \
    librosa \
    moviepy==2.2.1 \
    av \
    regex \
    soundfile \
    wget==3.2

# Clone SkyReels-V3 repository
RUN git clone https://github.com/SkyworkAI/SkyReels-V3.git /app/SkyReels-V3

COPY handler.py /app/handler.py

CMD ["python3", "-u", "handler.py"]
