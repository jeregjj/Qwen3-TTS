# SAP AI Core compliant Dockerfile for Qwen3-TTS
# Base: PyTorch 2.3.0 with CUDA 12.1 support for Python 3.10
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ffmpeg \
    git \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Create model cache directory
RUN mkdir -p /mnt/models/.cache

# Environment variables for model caching
ENV HF_HOME=/mnt/models/.cache \
    TRANSFORMERS_CACHE=/mnt/models/.cache/transformers \
    HF_DATASETS_CACHE=/mnt/models/.cache/datasets \
    HUGGINGFACE_HUB_CACHE=/mnt/models/.cache/hub \
    TORCH_HOME=/mnt/models/.cache/torch \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_ID=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    DEVICE=cuda:0 \
    PORT=9001

# Copy requirements and install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install pre-compiled flash-attn wheel (avoids 30+ min source build)
# This wheel is for CUDA 12.2 + PyTorch 2.3 + Python 3.10
RUN pip install --no-cache-dir https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Copy application source code
COPY src/ /app/src/

# SAP AI Core permissions (non-root UID compatibility)
RUN chgrp -R nogroup /app && chmod -R 770 /app && \
    chgrp -R nogroup /mnt/models && chmod -R 770 /mnt/models

# Expose KServe port
EXPOSE 9001

# Health check for KServe (uses correct endpoint path)
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:9001/v1/models/tts:health || exit 1

# Run Flask app with gunicorn
# - 1 worker (GPU model is not fork-safe)
# - 1 thread (PyTorch CUDA not thread-safe for inference)
# - 300s timeout (TTS generation can be slow for long texts)
# - preload to load model once before forking (if workers > 1 in future)
CMD ["gunicorn", "--bind", "0.0.0.0:9001", "--workers", "1", "--threads", "1", "--timeout", "300", "--preload", "src.serve:app"]
