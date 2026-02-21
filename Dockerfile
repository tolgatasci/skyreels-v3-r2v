FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/root/.cache/huggingface

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    git wget curl ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Upgrade PyTorch to 2.6+ (need add_safe_globals + torch.int1)
RUN pip install --no-cache-dir --upgrade \
    "torch>=2.6.0" \
    "torchvision>=0.21.0" \
    --index-url https://download.pytorch.org/whl/cu124

# Verify torch version
RUN python3 -c "import torch; print(f'PyTorch {torch.__version__}'); assert hasattr(torch.serialization, 'add_safe_globals'), 'add_safe_globals missing!'; print('add_safe_globals OK')"

# Skip flash-attn (source build takes 1h+), use PyTorch SDPA instead
ENV ATTN_BACKEND=sdpa

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
    "numpy>=1.23.5,<2" \
    einops \
    easydict \
    ftfy \
    tqdm \
    opencv-python-headless \
    kornia \
    omegaconf \
    torchao \
    pyloudnorm \
    librosa \
    moviepy==2.2.1

# Verify diffusers loads correctly
RUN python3 -c "from diffusers import DiffusionPipeline; print('diffusers OK')"

# Clone SkyReels-V3 repository
RUN git clone https://github.com/SkyworkAI/SkyReels-V3.git /app/SkyReels-V3

COPY handler.py /app/handler.py

CMD ["python3", "-u", "handler.py"]
