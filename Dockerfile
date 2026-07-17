# Use a standard PyTorch base image
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install standard dependencies
RUN pip install --no-cache-dir fastapi uvicorn pydantic soundfile qwen-tts

# Directly install the pre-compiled flash-attn wheel to prevent source building
RUN pip install --no-cache-dir https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

COPY app.py /app/app.py

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]