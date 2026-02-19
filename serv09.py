import asyncio
import websockets
import json
import numpy as np
import torch
from faster_whisper import WhisperModel

# --- CONFIG ---
PORT = 8765
MODEL_ID = "deepdml/faster-whisper-large-v3-turbo-ct2"
# Pause threshold in seconds (Adjust this: 0.8 is good for natural speech)
PAUSE_THRESHOLD = 0.8 

print("Loading Model...")
model = WhisperModel(MODEL_ID, device="cuda", compute_type="float16")

# VAD for accurate pause detection
vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', onnx=True)
print("✅ Server Ready (Line-by-Line Mode)")

INDIAN_PROMPT = "Indian English speaker, clear transcript, no hallucinations."

async def handle_stream(websocket):
    print("--> Client Connected")
    audio_buffer = np.array([], dtype=np.float32)
    SAMPLE_RATE = 16000
    silence_counter = 0 # seconds of consecutive silence
    
    try:
        async for message in websocket:
            chunk = np.frombuffer(message, dtype=np.float32)
            audio_buffer = np.concatenate((audio_buffer, chunk))
            
            # 1. Check for silence in the current chunk (VAD)
            chunk_tensor = torch.tensor(chunk)
            speech_prob = vad_model(chunk_tensor, SAMPLE_RATE).item()
            
            # Each chunk from client is ~0.064s (1024 samples / 16000)
            if speech_prob < 0.3:
                silence_counter += (len(chunk) / SAMPLE_RATE)
            else:
                silence_counter = 0

            # 2. If we have enough audio to process (e.g., 1 second accumulated)
            if len(audio_buffer) >= (SAMPLE_RATE * 1.0):
                segments, _ = model.transcribe(
                    audio_buffer,
                    beam_size=1,
                    initial_prompt=INDIAN_PROMPT,
                    vad_filter=True
                )
                text = " ".join([s.text.strip() for s in segments])

                if text:
                    # 3. Detect if a pause just happened
                    is_final = silence_counter >= PAUSE_THRESHOLD
                    
                    await websocket.send(json.dumps({
                        "text": text,
                        "final": is_final
                    }))

                    # 4. If finalized, clear the buffer to start a fresh line
                    if is_final:
                        audio_buffer = np.array([], dtype=np.float32)
                        silence_counter = 0 
                
                # Keep small context if not final to prevent cutting words
                if not is_final and len(audio_buffer) > (SAMPLE_RATE * 2):
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
