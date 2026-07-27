---
name: deploy-hf-model-to-aicore
description: Use when deploying any HuggingFace model to SAP AI Core for inference. Covers Dockerfile patterns, ServingTemplate configuration, Flask server setup, GPU compatibility, and common pitfalls like CUDA fork errors, FlashAttention compatibility, and model file downloads.
---

# Deploy HuggingFace Models to SAP AI Core

## Overview

This skill provides a complete guide for deploying open-source HuggingFace models to SAP AI Core for inference. It covers the entire pipeline from Dockerfile creation to production deployment, including GPU compatibility, KServe integration, and OAuth-authenticated testing.

**Core principle:** SAP AI Core uses KServe for model serving. Your container must expose specific endpoints, handle GPU constraints, and comply with AI Core's security requirements.

## When to Use

Use this skill when:
- Deploying any HuggingFace model (LLM, TTS, image, etc.) to SAP AI Core
- Building a Flask/FastAPI inference server for AI Core
- Troubleshooting AI Core deployment failures
- Setting up GPU-accelerated inference on AI Core

Do NOT use when:
- Deploying to other cloud platforms (AWS SageMaker, GCP Vertex AI)
- Using AI Core's built-in foundation models
- Training models on AI Core (use WorkflowTemplate instead)

## Quick Reference

| Component | Pattern |
|-----------|---------|
| Base Image | `pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel` |
| Port | 9001 (KServe standard) |
| Health endpoint | `/v1/models/{model}:health` |
| Predict endpoint | `/v1/models/{model}:predict` |
| Resource plan | `infer.l` (GPU), `infer.s` (CPU), `starter` (dev) |
| Attention impl | `sdpa` (safe), `flash_attention_2` (Ampere+ only) |
| Gunicorn | `--workers 1 --threads 1` (NO `--preload`) |

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    SAP AI Core                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              KServe Predictor                        │    │
│  │  ┌─────────────────────────────────────────────┐    │    │
│  │  │         Your Docker Container                │    │    │
│  │  │  ┌─────────────────────────────────────┐    │    │    │
│  │  │  │   Flask/Gunicorn Server (:9001)     │    │    │    │
│  │  │  │   - /v1/models/tts:health           │    │    │    │
│  │  │  │   - /v1/models/tts:predict          │    │    │    │
│  │  │  │   - /v1/models/tts:info             │    │    │    │
│  │  │  └─────────────────────────────────────┘    │    │    │
│  │  │  ┌─────────────────────────────────────┐    │    │    │
│  │  │  │   HuggingFace Model (GPU)           │    │    │    │
│  │  │  │   - Loaded at startup               │    │    │    │
│  │  │  │   - Cached in /mnt/models/.cache    │    │    │    │
│  │  │  └─────────────────────────────────────┘    │    │    │
│  │  └─────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## 1. Dockerfile Template

```dockerfile
# SAP AI Core compliant Dockerfile for HuggingFace models
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel

WORKDIR /app

# System dependencies (adjust based on your model's needs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ffmpeg \
    git \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Create cache directories
RUN mkdir -p /mnt/models/.cache /tmp/numba_cache /tmp/matplotlib

# Environment variables for HuggingFace caching
ENV HF_HOME=/mnt/models/.cache \
    TRANSFORMERS_CACHE=/mnt/models/.cache/transformers \
    HF_DATASETS_CACHE=/mnt/models/.cache/datasets \
    HUGGINGFACE_HUB_CACHE=/mnt/models/.cache/hub \
    TORCH_HOME=/mnt/models/.cache/torch \
    NUMBA_CACHE_DIR=/tmp/numba_cache \
    MPLCONFIGDIR=/tmp/matplotlib \
    PYTHONUNBUFFERED=1 \
    MODEL_ID=your-org/your-model \
    DEVICE=cuda:0 \
    PORT=9001

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# CRITICAL: Pre-download model files that won't download correctly at runtime
# This is especially important for models with subdirectories (tokenizers, etc.)
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('your-org/your-model', \
        allow_patterns=['speech_tokenizer/*', 'tokenizer/*'], \
        cache_dir='/mnt/models/.cache')"

# Copy application code
COPY src/ /app/src/

# SAP AI Core permissions (REQUIRED - non-root UID compatibility)
RUN chgrp -R nogroup /app && chmod -R 770 /app && \
    chgrp -R nogroup /mnt/models && chmod -R 770 /mnt/models && \
    chmod -R 777 /tmp/numba_cache /tmp/matplotlib

# KServe port
EXPOSE 9001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:9001/v1/models/tts:health || exit 1

# CRITICAL: NO --preload flag (breaks CUDA)
CMD ["gunicorn", "--bind", "0.0.0.0:9001", "--workers", "1", "--threads", "1", "--timeout", "300", "src.serve:app"]
```

### Key Dockerfile Patterns

| Pattern | Why |
|---------|-----|
| `chgrp -R nogroup && chmod -R 770` | AI Core runs containers with arbitrary UIDs |
| `mkdir -p /mnt/models/.cache` | Standard mount point for model caching |
| Pre-download with `snapshot_download()` | Some model subdirectories don't download at runtime |
| `--workers 1 --threads 1` | GPU models aren't fork-safe or thread-safe |
| NO `--preload` | CUDA cannot reinitialize in forked processes |
| `--timeout 300` | AI models can take minutes for first inference |

## 2. requirements.txt Template

```txt
# Web framework
Flask==3.0.3
gunicorn==22.0.0
Werkzeug==3.0.3

# HuggingFace ecosystem
# CRITICAL: Pin transformers to 4.x (5.x requires PyTorch 2.5+)
transformers>=4.44.0,<5.0.0
accelerate>=0.33.0
huggingface-hub>=0.24.5

# Audio processing (if needed)
soundfile==0.12.1
librosa==0.10.2.post1
scipy==1.14.0

# Numerical computing
numpy>=1.26.4
einops>=0.8.0
safetensors>=0.4.4
```

## 3. Flask Server Template (serve.py)

```python
"""
HuggingFace Model Inference Server for SAP AI Core
KServe-compliant Flask application
"""

# CRITICAL: Set environment variables BEFORE imports
import os
os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/numba_cache')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')

import logging
import sys
import traceback
import torch
from flask import Flask, jsonify, request

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("inference-server")

# Configuration from environment
DEVICE = os.environ.get("DEVICE", "cuda:0")
MODEL_ID = os.environ.get("MODEL_ID", "your-org/your-model")
PORT = int(os.environ.get("PORT", "9001"))

# Global model
model = None
model_load_error = None

app = Flask(__name__)


def load_model():
    """Load model at startup."""
    global model, model_load_error
    
    logger.info(f"Loading model: {MODEL_ID} on {DEVICE}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    try:
        from transformers import AutoModel  # Or your specific model class
        
        model = AutoModel.from_pretrained(
            MODEL_ID,
            device_map=DEVICE,
            torch_dtype=torch.bfloat16,
            # CRITICAL: Use 'sdpa' for older GPUs, 'flash_attention_2' for Ampere+
            attn_implementation="sdpa",
        )
        
        logger.info("Model loaded successfully")
        model_load_error = None
        
    except Exception as e:
        model_load_error = str(e)
        logger.error(f"Failed to load model: {e}")
        logger.error(traceback.format_exc())
        model = None


@app.route("/v1/models/<model_name>:health", methods=["GET"])
def health_check(model_name):
    """Health check endpoint - returns 503 if model not loaded."""
    if model is None:
        return jsonify({
            "status": "unhealthy",
            "model_loaded": False,
            "error": model_load_error or "Model not initialized",
        }), 503
    
    return jsonify({
        "status": "healthy",
        "model_loaded": True,
        "device": DEVICE,
        "model_id": MODEL_ID,
    }), 200


@app.route("/v1/models/<model_name>:info", methods=["GET"])
def model_info(model_name):
    """Model information endpoint."""
    return jsonify({
        "model_id": MODEL_ID,
        "device": DEVICE,
        "model_loaded": model is not None,
    }), 200


@app.route("/v1/models/<model_name>:predict", methods=["POST"])
def predict(model_name):
    """Main inference endpoint."""
    if model is None:
        return jsonify({
            "error": f"Model not loaded: {model_load_error}",
        }), 503
    
    try:
        data = request.get_json(silent=True) or {}
        
        # Your inference logic here
        # result = model.generate(...)
        
        return jsonify({
            "result": "your_result",
            "request_id": f"req-{os.urandom(6).hex()}",
        }), 200
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        return jsonify({
            "error": str(e),
            "exception_type": type(e).__name__,
        }), 500


# Load model at module import time
load_model()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
```

## 4. ServingTemplate YAML

```yaml
apiVersion: ai.sap.com/v1alpha1
kind: ServingTemplate
metadata:
  name: your-model-serving
  annotations:
    scenarios.ai.sap.com/id: "scenario-your-model"
    scenarios.ai.sap.com/name: "Your Model Scenario"
    executables.ai.sap.com/name: "your-model-serving"
    executables.ai.sap.com/description: "Your model inference server"
  labels:
    scenarios.ai.sap.com/id: "scenario-your-model"
    executables.ai.sap.com/id: "your-model-serving"
    ai.sap.com/version: "1.0.0"
spec:
  inputs:
    parameters: []
  template:
    apiVersion: "serving.kserve.io/v1beta1"
    metadata:
      labels: |
        ai.sap.com/resourcePlan: infer.l
    spec: |
      predictor:
        minReplicas: 1
        maxReplicas: 1
        containers:
          - name: kserve-container
            image: your-registry/your-image:tag
            ports:
              - containerPort: 9001
                protocol: TCP
            env:
              - name: HF_HOME
                value: "/mnt/models/.cache"
              - name: MODEL_ID
                value: "your-org/your-model"
              - name: DEVICE
                value: "cuda:0"
            resources:
              limits:
                nvidia.com/gpu: "1"
                memory: "32Gi"
                cpu: "8"
              requests:
                nvidia.com/gpu: "1"
                memory: "16Gi"
                cpu: "4"
```

### YAML Indentation Rules

**CRITICAL:** The `spec: |` uses a literal block. Content must be indented exactly 6 spaces:

```yaml
    spec: |
      predictor:        # 6 spaces
        minReplicas: 1  # 8 spaces
```

## 5. Building and Pushing

```bash
# CRITICAL: Always build for linux/amd64 (AI Core runs on x86)
docker build --platform linux/amd64 -t your-registry/your-image:tag .

# Push to registry
docker push your-registry/your-image:tag
```

## 6. Testing Script Template

Create a `.env` file:
```
AICORE_DEPLOYMENT_URL=https://api.ai.xxx.ml.hana.ondemand.com/v2/inference/deployments/xxx
AICORE_AUTH_URL=https://xxx.authentication.xxx.hana.ondemand.com
AICORE_CLIENT_ID=sb-xxx
AICORE_CLIENT_SECRET=xxx
AICORE_RESOURCE_GROUP=default
```

Test endpoints:
```bash
# Get OAuth token
TOKEN=$(curl -s -X POST "$AICORE_AUTH_URL/oauth/token?grant_type=client_credentials" \
  -H "Authorization: Basic $(echo -n "$AICORE_CLIENT_ID:$AICORE_CLIENT_SECRET" | base64)" \
  -d "grant_type=client_credentials" | jq -r '.access_token')

# Test health
curl -H "Authorization: Bearer $TOKEN" \
     -H "AI-Resource-Group: $AICORE_RESOURCE_GROUP" \
     "$AICORE_DEPLOYMENT_URL/v1/models/your-model:health"

# Test predict
curl -X POST -H "Authorization: Bearer $TOKEN" \
     -H "AI-Resource-Group: $AICORE_RESOURCE_GROUP" \
     -H "Content-Type: application/json" \
     -d '{"input": "test"}' \
     "$AICORE_DEPLOYMENT_URL/v1/models/your-model:predict"
```

## Common Errors and Fixes

### 1. "name 'torch' is not defined"
**Cause:** transformers 5.x requires PyTorch 2.5+
**Fix:** Pin `transformers>=4.44.0,<5.0.0`

### 2. "Can't load feature extractor... preprocessor_config.json"
**Cause:** HuggingFace subdirectory files not downloaded
**Fix:** Pre-download in Dockerfile:
```dockerfile
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('model-id', allow_patterns=['tokenizer/*'], cache_dir='/mnt/models/.cache')"
```

### 3. "ErrImagePull: no match for platform"
**Cause:** Image built on ARM Mac, AI Core needs linux/amd64
**Fix:** `docker build --platform linux/amd64`

### 4. "Cannot re-initialize CUDA in forked subprocess"
**Cause:** gunicorn `--preload` flag
**Fix:** Remove `--preload` from CMD

### 5. "FlashAttention only supports Ampere GPUs or newer"
**Cause:** AI Core uses V100/T4 (pre-Ampere)
**Fix:** Use `attn_implementation="sdpa"` instead of `"flash_attention_2"`

### 6. Health returns 503, deployment shows "RevisionMissing"
**Cause:** Container failing to start (check logs)
**Fix:** Check deployment details for exact error message

## Resource Plans

| Plan | GPU | Memory | Use Case |
|------|-----|--------|----------|
| `starter` | None | 4Gi | Development, CPU-only |
| `infer.s` | None | 8Gi | Small models, CPU |
| `infer.m` | 1x T4 | 16Gi | Medium models |
| `infer.l` | 1x V100 | 32Gi | Large models |

**Note:** V100/T4 GPUs do NOT support FlashAttention 2.

## External References

- [SAP AI Core Documentation](https://help.sap.com/docs/ai-core)
- [KServe Predictor Spec](https://kserve.github.io/website/latest/modelserving/v1beta1/serving_runtime/)
- [HuggingFace Hub Documentation](https://huggingface.co/docs/huggingface_hub)
- [PyTorch CUDA Compatibility](https://pytorch.org/get-started/locally/)
- [FlashAttention Requirements](https://github.com/Dao-AILab/flash-attention#installation-and-features)

## Checklist

Before deploying:
- [ ] Dockerfile uses `--platform linux/amd64` for build
- [ ] `chgrp -R nogroup && chmod -R 770` applied
- [ ] No `--preload` flag in gunicorn CMD
- [ ] `attn_implementation="sdpa"` (unless you KNOW you have Ampere+ GPUs)
- [ ] Model subdirectories pre-downloaded if needed
- [ ] transformers pinned to 4.x
- [ ] Health endpoint returns 503 when model not loaded
- [ ] ServingTemplate YAML has correct indentation (6 spaces in spec block)
- [ ] Resource plan matches your GPU/memory needs
