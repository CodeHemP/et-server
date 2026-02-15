from fastapi import FastAPI, UploadFile
from faster_whisper import WhisperModel
import tempfile
import os

app = FastAPI()

# Load English-only model
model = WhisperModel(
    "medium.en",
    device="cuda",
    compute_type="float16",
    download_root="/workspace/models"
)

@app.post("/transcribe")
async def transcribe(file: UploadFile):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        segments, _ = model.transcribe(
            tmp_path,
            beam_size=1,
            temperature=0.0
        )

        text = " ".join(seg.text.strip() for seg in segments)

        return {"text": text.strip()}

    finally:
        os.remove(tmp_path)
