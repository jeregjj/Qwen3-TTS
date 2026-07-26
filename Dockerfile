# SAP AI Core compliant Dockerfile for Qwen3-TTS
# Base: PyTorch 2.4.1 with CUDA 12.4 support for Python 3.10
# Note: transformers>=4.44.0 requires PyTorch >= 2.4
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ffmpeg \
    git \
    libsndfile1 \
    sox \
    && rm -rf /var/lib/apt/lists/*

# Create model cache directory and numba cache directory
RUN mkdir -p /mnt/models/.cache /tmp/numba_cache /tmp/matplotlib

# Environment variables for model caching and numba
ENV HF_HOME=/mnt/models/.cache \
    TRANSFORMERS_CACHE=/mnt/models/.cache/transformers \
    HF_DATASETS_CACHE=/mnt/models/.cache/datasets \
    HUGGINGFACE_HUB_CACHE=/mnt/models/.cache/hub \
    TORCH_HOME=/mnt/models/.cache/torch \
    NUMBA_CACHE_DIR=/tmp/numba_cache \
    MPLCONFIGDIR=/tmp/matplotlib \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_ID=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    DEVICE=cuda:0 \
    PORT=9001

# Copy requirements and install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install pre-compiled flash-attn wheel (avoids 30+ min source build)
# This wheel is for CUDA 12 + PyTorch 2.4 + Python 3.11
RUN pip install --no-cache-dir https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1+cu12torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

# Copy application source code and local qwen_tts package
COPY src/ /app/src/
COPY qwen_tts/ /app/qwen_tts/

# SAP AI Core permissions (non-root UID compatibility)
# Also set permissions on tmp cache directories for numba/matplotlib
RUN chgrp -R nogroup /app && chmod -R 770 /app && \
    chgrp -R nogroup /mnt/models && chmod -R 770 /mnt/models && \
    chmod -R 777 /tmp/numba_cache /tmp/matplotlib

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
