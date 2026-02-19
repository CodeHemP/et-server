import asyncio
import websockets
import json
import numpy as np
from faster_whisper import WhisperModel

# --- CONFIG ---
PORT = 8765
# This is the fastest SOTA model (Large accuracy, Turbo speed)
MODEL_ID = "deepdml/faster-whisper-large-v3-turbo-ct2"

print("Loading Model... (4090 detected)")
# float16 is native to 4090 and extremely fast
model = WhisperModel(MODEL_ID, device="cuda", compute_type="float16")
print("✅ Transcription Server Ready!")

# Optimized prompt for Indian English nuances
INDIAN_PROMPT = "This is a recording of an Indian speaker. Use Indian English spellings and names correctly."

async def handle_stream(websocket):
    print("--> Client Connected")
    audio_buffer = np.array([], dtype=np.float32)
    SAMPLE_RATE = 16000
    
    try:
        async for message in websocket:
            # 1. Receive raw audio bytes
            chunk = np.frombuffer(message, dtype=np.float32)
            audio_buffer = np.concatenate((audio_buffer, chunk))
            
            # 2. Process every 2 seconds
            if len(audio_buffer) >= (SAMPLE_RATE * 2):
                
                # transcribe
                segments, info = model.transcribe(
                    audio_buffer,
                    beam_size=1,            # 1 = Blazing fast, 5 = More accurate
                    language="en",
                    initial_prompt=INDIAN_PROMPT,
                    vad_filter=True,        # Prevents hallucinations in silence
                    vad_parameters=dict(min_silence_duration_ms=500)
                )
                
                # Combine results
                text = " ".join([s.text.strip() for s in segments])
                
                if text:
                    await websocket.send(json.dumps({"text": text}))
                
                # 3. Rolling window: Keep last 1 second to prevent word cutting
                audio_buffer = audio_buffer[-SAMPLE_RATE:]

    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("--> Client Disconnected")

async def main():
    async with websockets.serve(handle_stream, "0.0.0.0", PORT):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
