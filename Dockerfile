# Utilizing an optimized NVIDIA PyTorch base image containing Python 3.12 capabilities
FROM nvcr.io/nvidia/pytorch:24.03-py3

WORKDIR /app

# Install system dependencies required for processing audio files
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install standard application dependencies
RUN pip install --no-cache-dir fastapi uvicorn pydantic soundfile

# Install the primary qwen-tts module
RUN pip install -U qwen-tts

# Build FlashAttention 2 without isolation to fit GPU architecture
RUN pip install -U flash-attn --no-build-isolation

# Copy application script
COPY app.py /app/app.py

# Expose target port for SAP AI Core tracking routing
EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]