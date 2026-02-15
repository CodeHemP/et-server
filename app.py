from fastapi import FastAPI, WebSocket
from faster_whisper import WhisperModel
import numpy as np
import tempfile
import scipy.io.wavfile as wav
import os

app = FastAPI()

model = WhisperModel(
    "medium",
    device="cuda",
    compute_type="float16",
    download_root="/workspace/models"
)

SAMPLE_RATE = 16000
WINDOW_SECONDS = 5
STEP_SECONDS = 1

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected")

    audio_buffer = np.zeros((0,), dtype=np.int16)

    try:
        while True:
            data = await websocket.receive_bytes()

            chunk = np.frombuffer(data, dtype=np.int16)
            audio_buffer = np.concatenate((audio_buffer, chunk))

            required_samples = WINDOW_SECONDS * SAMPLE_RATE
            step_samples = STEP_SECONDS * SAMPLE_RATE

            if len(audio_buffer) >= required_samples:
                window = audio_buffer[-required_samples:]

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    wav.write(tmp.name, SAMPLE_RATE, window)
                    tmp_path = tmp.name

                segments, _ = model.transcribe(
                    tmp_path,
                    beam_size=1,
                    temperature=0.0
                )

                os.remove(tmp_path)

                text = " ".join(seg.text.strip() for seg in segments)

                if text:
                    await websocket.send_text(text)

                audio_buffer = audio_buffer[step_samples:]

    except Exception as e:
        print("Client disconnected:", e)
