# Qwen3-TTS Deployment Troubleshooting Guide

> **Last Updated**: 2026-07-27  
> **Scope**: Qwen3-TTS deployment to SAP AI Core via KServe  
> **Audience**: DevOps engineers, deployment specialists, support staff

---

## Table of Contents

1. [PyTorch/Transformers Version Incompatibility](#1-pytorchtransformers-version-incompatibility)
2. [Missing HuggingFace Config Files](#2-missing-huggingface-config-files)
3. [Docker Platform Architecture Mismatch](#3-docker-platform-architecture-mismatch)
4. [API Method Name Mismatch](#4-api-method-name-mismatch)
5. [CUDA Forking Subprocess Error](#5-cuda-forking-subprocess-error)
6. [FlashAttention GPU Compatibility](#6-flashattention-gpu-compatibility)
7. [Quick Reference](#quick-reference)

---

## 1. PyTorch/Transformers Version Incompatibility

### Symptoms

```
RuntimeError: name 'torch' is not defined
ImportError: cannot import name 'torch' from transformers
AttributeError: module 'transformers' has no attribute 'PreTrainedModel'
```

### Root Cause

**transformers 5.x dropped support for PyTorch 2.3.x and 2.4.x**, requiring PyTorch 2.5.0+. However, the base Docker image uses PyTorch 2.4.1 (provided by `pytorch:2.4.1-cuda12.4-cudnn9-devel`). This version mismatch causes transformers to fail at import time or raise cryptic torch-related errors.

The issue manifests when:
- Installing transformers without pinning the version
- Upgrading to transformers 5.0+ accidentally
- Using a base image with PyTorch 2.4 or earlier

### Root Code Location

- **File**: `/Users/I772784/FDE/QWEN/Qwen3-TTS/requirements.txt` (line 17)
- **Issue**: Version constraint not specified correctly

### Fix

**Pin transformers to 4.x series**:

```diff
# File: requirements.txt
- transformers>=4.44.0,<5.0.0
+ transformers>=4.44.0,<5.0.0
```

**Verify the fix**:

```bash
# In the Docker container
python -c "import transformers; print(transformers.__version__)"
# Expected output: 4.44.x, 4.45.x, etc. (NOT 5.x)

python -c "import torch; print(torch.__version__)"
# Expected output: 2.4.1
```

### Production Checklist

- [ ] Verify `requirements.txt` contains `transformers>=4.44.0,<5.0.0`
- [ ] Rebuild Docker image after any requirements change
- [ ] Test with `docker run <image> python -c "from transformers import AutoModel; print('OK')"`
- [ ] Pin to exact minor version in prod: `transformers==4.44.0`

### Prevention

- Never use `transformers>=4.44.0` without the upper bound constraint
- Test base image compatibility before deployment
- Use `pip freeze` to lock exact versions in lock files

---

## 2. Missing HuggingFace Config Files

### Symptoms

```
OSError: Can't load feature extractor. Can't find 'preprocessor_config.json' in 'Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice'
FileNotFoundError: [Errno 2] No such file or directory: '...speech_tokenizer/preprocessor_config.json'
HFValidationError: Repo not found
```

### Root Cause

The HuggingFace Hub's `snapshot_download()` function doesn't always fetch all required files, particularly nested config files in subdirectories like `speech_tokenizer/preprocessor_config.json`. When the model loads at inference time, these files are missing from the cache.

This occurs because:
1. HuggingFace model repos have multiple configuration files in nested directories
2. Partial downloads leave some files behind
3. Runtime doesn't fail until the specific file is accessed
4. Docker layer caching can hide the issue on rebuild

### Root Code Location

- **File**: `/Users/I772784/FDE/QWEN/Qwen3-TTS/Dockerfile` (lines 44-49)
- **Before Fix**: Did not explicitly pre-download nested config files

### Fix

**Pre-download all required files during Docker build**:

```dockerfile
# In Dockerfile, BEFORE copying application code
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice', \
        allow_patterns=['speech_tokenizer/*'], \
        cache_dir='/mnt/models/.cache')" && \
    echo 'Speech tokenizer pre-downloaded successfully' && \
    find /mnt/models/.cache -name 'speech_tokenizer' -type d -exec ls -la {} \;
```

**Verify the fix**:

```bash
# After docker build completes, inspect cache
docker run <image> find /mnt/models/.cache -name 'preprocessor_config.json' -type f

# Expected: One or more .json files in speech_tokenizer directory
# If empty: Pre-download failed, check build logs
```

### Production Checklist

- [ ] Pre-download explicitly runs in Dockerfile before CMD
- [ ] Build log shows "Speech tokenizer pre-downloaded successfully"
- [ ] `find` command returns at least one `.json` file
- [ ] Test inference in container to confirm config loads
- [ ] Do NOT rely on lazy download at runtime

### Prevention

- Always test model loading in Dockerfile before deployment
- For any model with subdirectories, explicitly pre-download those patterns
- Verify cached files exist: `docker run <image> ls /mnt/models/.cache/hub/`
- Document which HuggingFace snapshots are pre-baked

---

## 3. Docker Platform Architecture Mismatch

### Symptoms

```
ErrImagePull: image platform does not match (image is 'linux/arm64' but need 'linux/amd64')
CrashLoopBackOff: Exec format error
exec user process caused: no such file or directory
node.kubernetes.io/not-ready: Ready:False
```

### Root Cause

Docker images built on ARM-based machines (Apple Silicon, ARM-based Linux) default to `linux/arm64` architecture. However, SAP AI Core runs on `linux/amd64` x86-64 architecture. When the container runtime tries to execute ARM binaries on x86-64 hardware, the kernel refuses.

This happens when:
1. Developer builds on Apple Silicon Mac without specifying platform
2. Pushing multi-platform images without explicit platform selection
3. Pre-compiled wheels (e.g., flash-attn) have wrong architecture for target

### Root Code Location

- **File**: `/Users/I772784/FDE/QWEN/Qwen3-TTS/Dockerfile` (line 4)
- **Issue**: Base image selected without explicit platform

### Fix

**Build for x86-64 explicitly**:

```bash
# Method 1: Use --platform flag
docker build --platform linux/amd64 -t qwen3-tts:latest .

# Method 2: Set buildx builder
docker buildx build --platform linux/amd64 -t qwen3-tts:latest .

# Method 3: For Apple Silicon, use native builder with platform override
DOCKER_BUILDKIT=1 docker build \
    --platform=linux/amd64 \
    -t aibus-dev.common.repositories.cloud.sap/i343697/qwen3-tts:0.0.1 .
```

**Verify platform of built image**:

```bash
docker inspect qwen3-tts:latest | jq '.Os, .Architecture'
# Expected output:
# "linux"
# "amd64"
```

### Production Checklist

- [ ] All builds use `--platform linux/amd64`
- [ ] Docker inspect confirms `"Architecture": "amd64"`
- [ ] Base image supports multi-platform (pytorch:2.4.1-cuda12.4 does)
- [ ] Flash-attn wheel is x86-64: filename contains `x86_64`
- [ ] Test on x86-64 hardware or use emulation

### Prevention

- Add `--platform linux/amd64` to all build commands in CI/CD
- Document in deployment scripts
- Fail CI if platform != amd64
- For local development on Apple Silicon, always build with platform flag

---

## 4. API Method Name Mismatch

### Symptoms

```
AttributeError: 'Qwen3TTSModel' object has no attribute 'synthesize'
AttributeError: 'Qwen3TTSModel' object has no attribute 'tts'
TypeError: generate_custom_voice() missing required positional argument 'speaker'
```

### Root Cause

The Qwen3-TTS model API provides a specific method name: `generate_custom_voice()`. Older documentation, examples, or similar models use different method names like `synthesize()`, `tts()`, or `forward()`. Calling the wrong method raises an AttributeError.

The correct method signature is:
```python
wavs, sr = model.generate_custom_voice(
    text: str,
    language: str,
    speaker: str,
    instruct: Optional[str] = None
)
```

### Root Code Location

- **File**: `/Users/I772784/FDE/QWEN/Qwen3-TTS/src/serve.py` (lines 263-268)
- **Correct implementation**: Already uses `generate_custom_voice()`

### Fix

**Use the correct API method**:

```python
# WRONG (will fail)
wavs, sr = model.synthesize(text=text, speaker=speaker)
wavs, sr = model.tts(text)

# CORRECT
wavs, sr = model.generate_custom_voice(
    text=text,
    language=language,
    speaker=speaker,
    instruct=instruct if instruct else None
)
```

**Verify available methods**:

```python
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained('Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice')

# List all callable methods
methods = [m for m in dir(model) if callable(getattr(model, m)) and not m.startswith('_')]
print(methods)

# Should include: generate_custom_voice, from_pretrained, to, etc.
```

### Production Checklist

- [ ] Inference code uses `generate_custom_voice()`
- [ ] All required parameters passed: text, language, speaker
- [ ] Optional parameter `instruct` is conditional (None if not provided)
- [ ] Test with actual model load to verify method exists
- [ ] Document API in code comments for future maintainers

### Prevention

- Test model API locally before building Docker image
- Use official documentation for method signatures
- Add type hints in code for IDE autocomplete
- Include integration tests that call the actual model method

---

## 5. CUDA Forking Subprocess Error

### Symptoms

```
RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method
RuntimeError: CUDA memory is already allocated
CUDA out of memory: tried to allocate 0.00 MiB (GPU 0; 15.90 GiB total capacity; 0 B free)
```

### Root Cause

PyTorch's CUDA runtime cannot safely fork processes (used by `multiprocessing` and `gunicorn --preload`). When gunicorn forks worker processes with `--preload`, the CUDA context is copied to child processes, causing state corruption.

The root cause:
- `--preload` flag loads the app in the master process
- Master process initializes CUDA context
- Child processes inherit corrupted CUDA state
- First CUDA operation in child fails

### Root Code Location

- **File**: `/Users/I772784/FDE/QWEN/Qwen3-TTS/Dockerfile` (line 73)
- **Issue**: Original command used `--preload` flag (now removed)

### Fix

**Remove `--preload` from gunicorn command**:

```diff
# File: Dockerfile
- CMD ["gunicorn", "--bind", "0.0.0.0:9001", "--workers", "1", "--threads", "1", "--timeout", "300", "--preload", "src.serve:app"]
+ CMD ["gunicorn", "--bind", "0.0.0.0:9001", "--workers", "1", "--threads", "1", "--timeout", "300", "src.serve:app"]
```

**Why this works**:
- App loads in each worker process independently
- CUDA context initialized fresh in each worker
- No shared state between workers
- Since we use `--workers 1`, only one process loads the model anyway

**Verify the fix**:

```bash
# Test locally
docker run --gpus all -p 9001:9001 qwen3-tts:latest

# Should start without CUDA errors
# Logs should show model loading, no RuntimeError
```

### Production Checklist

- [ ] `--preload` flag removed from CMD
- [ ] `--workers 1` used (TTS models not fork-safe)
- [ ] `--threads 1` used (PyTorch CUDA not thread-safe for inference)
- [ ] Test deployment in AI Core
- [ ] Monitor logs for any CUDA errors on first inference

### Prevention

- Never use `--preload` with GPU models
- Document why workers=1, threads=1 in Dockerfile comments
- Use process managers that support spawn (not default fork)
- Test multiprocessing with CUDA locally before deployment

---

## 6. FlashAttention GPU Compatibility

### Symptoms

```
NotImplementedError: FlashAttention only supports Ampere GPUs or newer
RuntimeError: Compiler not found: CUDA Compiler Collection (NVCC) not found
```

### Root Cause

FlashAttention 2 requires NVIDIA GPUs with Ampere architecture or newer (A100, A40, A30, RTX 30 series, H100, etc.). Older GPUs like V100, T4, and P100 don't have the hardware features (TensorFloat-32, structured matrix operations) that FlashAttention depends on.

SAP AI Core may use older GPU node pools. When the inference container tries to use `attn_implementation="flash_attention_2"`, the GPU hardware doesn't support it.

### Root Code Location

- **File**: `/Users/I772784/FDE/QWEN/Qwen3-TTS/Dockerfile` (line 40)
- **File**: `/Users/I772784/FDE/QWEN/Qwen3-TTS/src/serve.py` (line 198)
- **Issue**: Hard-coded to use flash_attention_2

### Fix

**Use fallback attention implementation**:

```python
# File: src/serve.py

# Method 1: Try flash_attention, fall back to SDPA
try:
    model = Qwen3TTSModel.from_pretrained(
        model_path,
        device_map=DEVICE,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
except RuntimeError as e:
    if "Ampere" in str(e) or "NVCC" in str(e):
        logger.warning(f"FlashAttention not supported, falling back to SDPA: {e}")
        model = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map=DEVICE,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",  # Scaled Dot-Product Attention
        )
    else:
        raise

# Method 2: Detect GPU and choose implementation
import torch

def get_attention_impl():
    if not torch.cuda.is_available():
        return "eager"
    
    # Get GPU capability
    device_id = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device_id)
    major_version = capability[0]
    
    if major_version >= 8:  # Ampere or newer
        return "flash_attention_2"
    else:
        return "sdpa"  # V100, T4, P100 use SDPA

model = Qwen3TTSModel.from_pretrained(
    model_path,
    device_map=DEVICE,
    dtype=torch.bfloat16,
    attn_implementation=get_attention_impl(),
)
```

**Dockerfile change (optional—install flash-attn conditionally)**:

```dockerfile
# Instead of always installing:
RUN pip install --no-cache-dir https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1+cu12torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

# Use:
RUN python -c "import torch; major = torch.cuda.get_device_capability(0)[0] if torch.cuda.is_available() else 0; exit(0 if major >= 8 else 1)" && \
    pip install --no-cache-dir https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1+cu12torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl || \
    echo "FlashAttention not supported on this GPU, will use SDPA"
```

**Verify the fix**:

```bash
# Test locally on different GPUs
docker run --gpus all qwen3-tts:latest python -c "
from src.serve import load_model
model = load_model()
print('Model loaded with chosen attention implementation')
"
```

### Production Checklist

- [ ] Code detects GPU type or handles flash_attention errors
- [ ] Fallback to SDPA implemented and tested
- [ ] Deploy and test on AI Core (may use older GPUs)
- [ ] Monitor logs for attention implementation chosen
- [ ] Document GPU requirements in README

### Prevention

- Always provide fallback attention implementations
- Don't hard-code flash_attention for production deployments
- Test on multiple GPU types (A100, V100, T4)
- Document GPU constraints in deployment templates

---

## Quick Reference

### Build Commands

```bash
# Build for x86-64 (required for AI Core)
docker build --platform linux/amd64 -t qwen3-tts:latest .

# Build and push to registry
docker build --platform linux/amd64 \
    -t aibus-dev.common.repositories.cloud.sap/i343697/qwen3-tts:0.0.1 .
docker push aibus-dev.common.repositories.cloud.sap/i343697/qwen3-tts:0.0.1
```

### Test Locally

```bash
# Run inference server (requires GPU)
docker run --gpus all -p 9001:9001 qwen3-tts:latest

# Test health endpoint
curl -s http://localhost:9001/v1/models/tts:health | jq

# Test inference
curl -X POST http://localhost:9001/v1/models/tts:predict \
    -H "Content-Type: application/json" \
    -d '{"text": "Hello, this is a test.", "speaker": "Ryan"}'
```

### Verify Fixes

```bash
# Check transformers version
docker run qwen3-tts:latest python -c "import transformers; print(transformers.__version__)"
# Expected: 4.44.x (NOT 5.x)

# Check platform
docker inspect qwen3-tts:latest | jq '.Architecture'
# Expected: "amd64"

# Check config files
docker run qwen3-tts:latest find /mnt/models/.cache -name 'preprocessor_config.json'
# Expected: At least one file found

# Check gunicorn command
docker run qwen3-tts:latest grep "CMD.*gunicorn" /app/Dockerfile
# Should NOT contain "--preload"
```

### Common Deployment Issues

| Issue | Check | Solution |
|-------|-------|----------|
| Image pull fails | Platform mismatch | Use `--platform linux/amd64` |
| Model loads but inference fails | transformers version | Pin to `<5.0.0` |
| FileNotFoundError on config | Missing pre-download | Add snapshot_download() in Dockerfile |
| CUDA fork error | gunicorn --preload | Remove --preload, use --workers 1 |
| FlashAttention error | Old GPU | Use attn_implementation="sdpa" fallback |

### Deployment Status Tracking

```bash
# Monitor deployment
watch -n 10 'curl -s "$AI_API_URL/v2/lm/deployments/<deploymentId>" \
    -H "Authorization: Bearer $TOKEN" \
    -H "AI-Resource-Group: <resource-group>" | jq ".status"'

# View recent logs
curl -s "$AI_API_URL/v2/lm/deployments/<deploymentId>/logs" \
    -H "Authorization: Bearer $TOKEN" \
    -H "AI-Resource-Group: <resource-group>" | jq '.data.result.logs[-20:]'

# Test deployed endpoint
curl -X POST "$DEPLOYMENT_URL/v1/models/tts:predict" \
    -H "Authorization: Bearer $TOKEN" \
    -H "AI-Resource-Group: <resource-group>" \
    -H "Content-Type: application/json" \
    -d '{"text": "Test speech.", "speaker": "Ryan", "format": "base64"}'
```

---

## Contact & Escalation

For issues not covered in this guide:

1. Check deployment logs: `curl ... /deployments/<id>/logs`
2. Verify image runs locally: `docker run --gpus all <image>`
3. Check SAP AI Core resource availability
4. Review model HuggingFace documentation for API changes

---

**Document Version**: 1.0  
**Last Updated**: 2026-07-27  
**Status**: Production-Ready
