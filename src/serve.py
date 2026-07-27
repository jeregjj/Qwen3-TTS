"""
Qwen3-TTS Flask Server
GPU-only deployment with Flash Attention 2
SAP AI Core compliant
"""

# IMPORTANT: Set numba environment variables BEFORE any imports
# This prevents librosa/numba caching errors in containerized environments
import os
os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/numba_cache')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')

import base64
import io
import logging
import sys
import traceback
from datetime import datetime

import scipy.io.wavfile as wavfile
import torch
from flask import Flask, jsonify, request, Response

# Configure logging with detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("qwen3-tts")

# Configuration
DEVICE = os.environ.get("DEVICE", "cuda:0")
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
PORT = int(os.environ.get("PORT", "9001"))

# Available speakers and languages for Qwen3-TTS
AVAILABLE_SPEAKERS = ['Vivian', 'Serena', 'Uncle_Fu', 'Dylan', 'Eric', 'Ryan', 'Aiden', 'Ono_Anna', 'Sohee']
SUPPORTED_LANGUAGES = ['English', 'Chinese', 'Japanese', 'Korean', 'German', 'French', 'Russian', 'Portuguese', 'Spanish', 'Italian', 'Auto']

# Global model
model = None
model_load_error = None

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024  # 1MB max request size


def diagnose_hf_cache():
    """Diagnostic function to check HuggingFace cache state before model loading.

    This runs BEFORE model loading to show what's in the cache.
    Run again AFTER a failed load to see what got downloaded.
    """
    cache_dir = os.environ.get('HF_HOME', '/mnt/models/.cache')
    logger.info("=" * 60)
    logger.info("DIAGNOSTIC: HuggingFace Cache Analysis")
    logger.info(f"HF_HOME: {cache_dir}")
    logger.info(f"TRANSFORMERS_CACHE: {os.environ.get('TRANSFORMERS_CACHE', 'not set')}")
    logger.info(f"HUGGINGFACE_HUB_CACHE: {os.environ.get('HUGGINGFACE_HUB_CACHE', 'not set')}")
    logger.info(f"HF_HOME exists: {os.path.exists(cache_dir)}")

    # Check for model cache directory
    model_name = "Qwen--Qwen3-TTS-12Hz-1.7B-CustomVoice"
    model_cache = os.path.join(cache_dir, f'models--{model_name}')

    if not os.path.exists(model_cache):
        logger.info(f"Model cache does NOT exist at: {model_cache}")
        logger.info("Model will be downloaded on first load")
        logger.info("=" * 60)
        return

    logger.info(f"Model cache exists at: {model_cache}")

    # Find and analyze snapshots
    snapshots_dir = os.path.join(model_cache, 'snapshots')
    if not os.path.exists(snapshots_dir):
        logger.info("  No snapshots directory found")
        logger.info("=" * 60)
        return

    for snapshot in os.listdir(snapshots_dir):
        snapshot_path = os.path.join(snapshots_dir, snapshot)
        if not os.path.isdir(snapshot_path):
            continue

        logger.info(f"  Snapshot: {snapshot}")

        # List all top-level files/dirs in snapshot
        try:
            items = os.listdir(snapshot_path)
            logger.info(f"    Top-level items: {items}")
        except Exception as e:
            logger.error(f"    Error listing snapshot: {e}")
            continue

        # Check speech_tokenizer specifically - this is critical
        speech_tok_dir = os.path.join(snapshot_path, 'speech_tokenizer')
        if os.path.exists(speech_tok_dir):
            logger.info(f"    speech_tokenizer/ EXISTS")
            try:
                for f in os.listdir(speech_tok_dir):
                    fpath = os.path.join(speech_tok_dir, f)
                    if os.path.isfile(fpath):
                        size = os.path.getsize(fpath)
                        logger.info(f"      - {f} ({size} bytes)")
                    else:
                        logger.info(f"      - {f}/ (directory)")
            except Exception as e:
                logger.error(f"    Error listing speech_tokenizer: {e}")

            # Critical check for preprocessor_config.json
            preproc = os.path.join(speech_tok_dir, 'preprocessor_config.json')
            if os.path.exists(preproc):
                logger.info(f"    ✓ preprocessor_config.json EXISTS")
                # Show contents for debugging
                try:
                    with open(preproc, 'r') as f:
                        content = f.read()
                    logger.info(f"    Content: {content[:200]}...")
                except Exception as e:
                    logger.error(f"    Error reading preprocessor_config.json: {e}")
            else:
                logger.warning(f"    ✗ preprocessor_config.json MISSING - THIS IS THE PROBLEM!")
        else:
            logger.warning(f"    ✗ speech_tokenizer/ does NOT exist!")

    logger.info("=" * 60)


def load_model():
    """Load model at startup."""
    global model, model_load_error

    logger.info(f"Starting model load: {MODEL_ID} on {DEVICE}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    try:
        from qwen_tts import Qwen3TTSModel

        logger.info(f"Loading Qwen3TTSModel from {MODEL_ID}...")
        model = Qwen3TTSModel.from_pretrained(
            MODEL_ID,
            device_map=DEVICE,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",  # Use SDPA instead of flash_attention_2 for older GPUs
        )

        logger.info(f"Model loaded successfully on {DEVICE}")
        logger.info(f"Model type: {type(model).__name__}")
        model_load_error = None

    except Exception as e:
        model_load_error = str(e)
        logger.error(f"Failed to load model: {e}")
        logger.error(traceback.format_exc())
        model = None
        # Run diagnostics again after failure to see what was downloaded
        logger.info("Running post-failure diagnostics...")
        diagnose_hf_cache()


def validate_speaker(speaker: str) -> tuple:
    """Validate speaker parameter."""
    if speaker not in AVAILABLE_SPEAKERS:
        return False, f"Invalid speaker '{speaker}'. Available speakers: {AVAILABLE_SPEAKERS}"
    return True, ""


def validate_language(language: str) -> tuple:
    """Validate language parameter."""
    if language not in SUPPORTED_LANGUAGES:
        return False, f"Invalid language '{language}'. Supported languages: {SUPPORTED_LANGUAGES}"
    return True, ""


def generate_speech(text: str, speaker: str = "Vivian", language: str = "Auto", instruct: str = "") -> tuple:
    """Generate speech from text using Qwen3-TTS."""
    logger.info(f"Generating speech: text_len={len(text)}, speaker={speaker}, language={language}")

    inputs = None
    outputs = None

    try:
        # Generate audio using Qwen3-TTS CustomVoice model
        # generate_custom_voice returns (List[np.ndarray], sample_rate)
        wavs, sample_rate = model.generate_custom_voice(
            text=text,
            speaker=speaker,
            language=language,
            instruct=instruct if instruct else None,
        )

        # Get the first (and only) audio sample from the list
        audio = wavs[0]

        # Convert to WAV bytes
        buffer = io.BytesIO()
        wavfile.write(buffer, sample_rate, audio)
        buffer.seek(0)

        logger.info(f"Speech generated successfully: {len(audio)} samples, {sample_rate}Hz")
        return buffer.read(), sample_rate

    except Exception as e:
        logger.error(f"Speech generation failed: {e}")
        logger.error(traceback.format_exc())
        raise

    finally:
        # Clean up GPU memory
        if inputs is not None:
            del inputs
        if outputs is not None:
            del outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def handle_tts_request(data: dict, request_id: str) -> tuple:
    """Handle TTS prediction request."""
    logger.info(f"[{request_id}] Processing TTS request")

    if model is None:
        error_msg = f"Model not loaded. Error: {model_load_error}" if model_load_error else "Model not initialized"
        logger.error(f"[{request_id}] {error_msg}")
        return {"error": error_msg, "request_id": request_id}, 503

    if not data:
        logger.warning(f"[{request_id}] No JSON data provided")
        return {"error": "No JSON data provided", "request_id": request_id}, 400

    text = data.get("text") or data.get("input")
    if not text:
        logger.warning(f"[{request_id}] Missing 'text' or 'input' field")
        return {"error": "Missing 'text' or 'input' field", "request_id": request_id}, 400

    # Strip and validate text
    text = text.strip()
    if not text:
        logger.warning(f"[{request_id}] Empty text after stripping")
        return {"error": "Text cannot be empty or whitespace only", "request_id": request_id}, 400

    speaker = data.get("speaker", "Vivian")
    language = data.get("language", "Auto")
    instruct = data.get("instruct", "")
    output_format = data.get("format", "base64")

    logger.info(f"[{request_id}] Request params: speaker={speaker}, language={language}, format={output_format}, text_len={len(text)}")

    # Validate parameters
    valid, msg = validate_speaker(speaker)
    if not valid:
        logger.warning(f"[{request_id}] Invalid speaker: {speaker}")
        return {"error": msg, "available_speakers": AVAILABLE_SPEAKERS, "request_id": request_id}, 400

    valid, msg = validate_language(language)
    if not valid:
        logger.warning(f"[{request_id}] Invalid language: {language}")
        return {"error": msg, "supported_languages": SUPPORTED_LANGUAGES, "request_id": request_id}, 400

    try:
        audio_bytes, sample_rate = generate_speech(text, speaker, language, instruct)

        if output_format == "wav":
            logger.info(f"[{request_id}] Returning WAV response: {len(audio_bytes)} bytes")
            return Response(
                audio_bytes,
                mimetype="audio/wav",
                headers={"Content-Disposition": "attachment; filename=speech.wav"}
            ), 200

        # Default: base64 encoded response
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
        logger.info(f"[{request_id}] Returning base64 response: {len(audio_base64)} chars")

        return {
            "audio": audio_base64,
            "sample_rate": sample_rate,
            "format": "wav",
            "speaker": speaker,
            "language": language,
            "text_length": len(text),
            "request_id": request_id,
        }, 200

    except Exception as e:
        error_msg = f"TTS generation failed: {str(e)}"
        logger.error(f"[{request_id}] {error_msg}")
        logger.error(f"[{request_id}] {traceback.format_exc()}")
        return {"error": error_msg, "request_id": request_id, "exception_type": type(e).__name__}, 500


def generate_request_id() -> str:
    """Generate a unique request ID."""
    import uuid
    return f"req-{uuid.uuid4().hex[:12]}"


@app.route("/v1/models/tts:predict", methods=["POST"])
def tts_predict():
    """Main TTS prediction endpoint."""
    request_id = generate_request_id()
    logger.info(f"[{request_id}] POST /v1/models/tts:predict")

    try:
        data = request.get_json(silent=True) or {}
    except Exception as e:
        logger.error(f"[{request_id}] Failed to parse JSON: {e}")
        return jsonify({"error": "Invalid JSON", "request_id": request_id}), 400

    result, status_code = handle_tts_request(data, request_id)

    if isinstance(result, Response):
        return result, status_code

    return jsonify(result), status_code


@app.route("/v1/predict", methods=["POST"])
def legacy_predict():
    """Legacy endpoint for backward compatibility."""
    request_id = generate_request_id()
    logger.info(f"[{request_id}] POST /v1/predict (legacy)")

    try:
        data = request.get_json(silent=True) or {}
    except Exception as e:
        logger.error(f"[{request_id}] Failed to parse JSON: {e}")
        return jsonify({"error": "Invalid JSON", "request_id": request_id}), 400

    result, status_code = handle_tts_request(data, request_id)

    if isinstance(result, Response):
        return result, status_code

    return jsonify(result), status_code


@app.route("/v1/models/tts:health", methods=["GET"])
def health_check():
    """Health check endpoint - returns 503 if model not loaded."""
    logger.debug("GET /v1/models/tts:health")

    if model is None:
        logger.warning("Health check failed: model not loaded")
        return jsonify({
            "status": "unhealthy",
            "model_loaded": False,
            "device": DEVICE,
            "error": model_load_error or "Model not initialized",
        }), 503

    return jsonify({
        "status": "healthy",
        "model_loaded": True,
        "device": DEVICE,
        "model_id": MODEL_ID,
    }), 200


@app.route("/v1/models/tts:info", methods=["GET"])
def model_info():
    """Model information endpoint."""
    logger.debug("GET /v1/models/tts:info")

    return jsonify({
        "model_id": MODEL_ID,
        "device": DEVICE,
        "dtype": "bfloat16",
        "attn_implementation": "flash_attention_2",
        "model_loaded": model is not None,
        "available_speakers": AVAILABLE_SPEAKERS,
        "supported_languages": SUPPORTED_LANGUAGES,
        "sample_rate": 24000,
        "endpoints": {
            "predict": "/v1/models/tts:predict",
            "health": "/v1/models/tts:health",
            "info": "/v1/models/tts:info",
            "legacy": "/v1/predict",
        },
    }), 200


@app.errorhandler(400)
def bad_request(e):
    """Handle 400 errors."""
    logger.warning(f"400 Bad Request: {e}")
    return jsonify({
        "error": "Bad request",
        "message": str(e.description) if hasattr(e, 'description') else str(e),
    }), 400


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    logger.warning(f"404 Not Found: {request.path}")
    return jsonify({
        "error": "Endpoint not found",
        "path": request.path,
        "available_endpoints": [
            "/v1/models/tts:predict",
            "/v1/models/tts:health",
            "/v1/models/tts:info",
            "/v1/predict",
        ],
    }), 404


@app.errorhandler(413)
def request_too_large(e):
    """Handle 413 errors."""
    logger.warning(f"413 Request Too Large")
    return jsonify({
        "error": "Request too large",
        "max_size_bytes": app.config['MAX_CONTENT_LENGTH'],
    }), 413


@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors."""
    logger.error(f"500 Internal Server Error: {e}")
    logger.error(traceback.format_exc())
    return jsonify({
        "error": "Internal server error",
        "message": str(e) if app.debug else "An unexpected error occurred",
    }), 500


@app.errorhandler(Exception)
def handle_exception(e):
    """Handle all unhandled exceptions."""
    logger.error(f"Unhandled exception: {type(e).__name__}: {e}")
    logger.error(traceback.format_exc())
    return jsonify({
        "error": "Internal server error",
        "exception_type": type(e).__name__,
        "message": str(e),
    }), 500


# Load model at module import time (required for gunicorn)
logger.info("=" * 60)
logger.info("Qwen3-TTS Server Starting")
logger.info(f"Model ID: {MODEL_ID}")
logger.info(f"Device: {DEVICE}")
logger.info(f"Port: {PORT}")
logger.info("=" * 60)

# Run diagnostics before attempting model load
diagnose_hf_cache()

load_model()

logger.info("=" * 60)
logger.info(f"Model load complete. Status: {'SUCCESS' if model is not None else 'FAILED'}")
logger.info("=" * 60)


if __name__ == "__main__":
    logger.info(f"Starting Flask development server on port {PORT}")
    app.run(host="0.0.0.0", port=PORT, threaded=False)  # Single thread for GPU safety
