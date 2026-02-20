import asyncio
import json
import numpy as np
from fastapi import FastAPI, WebSocket
from faster_whisper import WhisperModel
from vllm import LLM, SamplingParams

app = FastAPI()

# 1. Initialize Engines
# Using Large-v3 for best English accuracy
asr = WhisperModel("large-v3", device="cuda", compute_type="float16")
# Using AWQ to save VRAM for TTS overhead later
mt = LLM(model="Qwen/Qwen2.5-7B-Instruct-AWQ", quantization="awq")
sampling_params = SamplingParams(temperature=0.1, max_tokens=100)

@app.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Professor connected.")

    while True:
        try:
            # Receive raw audio bytes from laptop
            data = await websocket.receive_bytes()
            audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

            # STAGE 1: Real-time Transcription
            # 'beam_size=1' is for speed; 'condition_on_previous_text' helps lecture context
            segments, _ = asr.transcribe(audio_np, beam_size=1, language="en")
            
            for segment in segments:
                eng_text = segment.text.strip()
                if not eng_text: continue

                # STAGE 2: Wait-k Prefix Translation (English -> Hindi)
                # We prompt the LLM to provide a 'simultaneous' translation
                prompt = f"<|im_start|>system\nYou are a simultaneous interpreter. Translate the following lecture fragment into natural Hindi. Maintain SOV grammar.<|im_end|>\n<|im_start|>user\n{eng_text}<|im_end|>\n<|im_start|>assistant\n"
                
                outputs = mt.generate([prompt], sampling_params)
                hindi_text = outputs[0].outputs[0].text.strip()

                # STAGE 3: Return Text immediately
                # (TTS will be added in the next step once text flow is verified)
                await websocket.send_json({
                    "original": eng_text,
                    "translated": hindi_text
                })

        except Exception as e:
            print(f"Connection closed: {e}")
            break

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
