import io
import torch
import soundfile as sf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from qwen_tts import Qwen3TTSModel

app = FastAPI(title="Qwen3-TTS Serving API on SAP AI Core")

# Global model placeholder
model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        # Load the model directly using the HuggingFace identifier
        model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
    except Exception as e:
        print(f"Error initializing Qwen3-TTS model: {e}")

class TTSRequest(BaseModel):
    text: str
    language: str = "Auto"
    speaker: str = "Vivian"
    instruct: str = ""

@app.post("/v1/predict")
async def predict(request: TTSRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized yet.")
    
    try:
        # Generate the voice arrays using the code logic from the README
        wavs, sr = model.generate_custom_voice(
            text=request.text,
            language=request.language,
            speaker=request.speaker,
            instruct=request.instruct if request.instruct else None
        )
        
        # Write array into an in-memory WAV file
        buffer = io.BytesIO()
        sf.write(buffer, wavs[0], sr, format="WAV")
        buffer.seek(0)
        
        return StreamingResponse(buffer, media_type="audio/wav")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")