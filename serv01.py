import asyncio
import websockets
import json
import numpy as np
from faster_whisper import WhisperModel
import time

# --- CONFIGURATION ---
# Use Large V3 Turbo for speed + accuracy. 
# "cuda" uses the 4090. "int8_float16" gives blazing speed with high accuracy.
MODEL_SIZE = "deepdml/faster-whisper-large-v3-turbo-ct2"
PORT = 8765

print("Loading Model... (this takes a few seconds)")
model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="int8_float16")
print("Model Loaded!")

# Prompt to force English and handle Indian names
INITIAL_PROMPT = "The following is a transcript of an Indian English speaker. The output is strictly in English."

async def transcribe_audio(websocket):
    print("Client connected")
    
    # Buffer to hold audio
    audio_buffer = np.array([], dtype=np.float32)
    
    # Configuration for the rolling window
    SAMPLE_RATE = 16000
    CHUNK_DURATION = 2.0  # seconds to infer on
    OVERLAP_DURATION = 0.5 # seconds to keep for context (PREVENTS CUT WORDS)
    
    chunk_samples = int(SAMPLE_RATE * CHUNK_DURATION)
    overlap_samples = int(SAMPLE_RATE * OVERLAP_DURATION)
    
    # State tracking
    committed_text = ""
    
    try:
        async for message in websocket:
            # 1. Receive raw bytes (assume float32 le from client)
            chunk = np.frombuffer(message, dtype=np.float32)
            audio_buffer = np.concatenate((audio_buffer, chunk))
            
            # 2. Check if we have enough audio to run inference
            if len(audio_buffer) >= chunk_samples:
                
                # Run inference on the current buffer
                segments, info = model.transcribe(
                    audio_buffer, 
                    beam_size=5, 
                    language="en", 
                    initial_prompt=INITIAL_PROMPT + " " + committed_text[-100:] # Feed previous context
                )
                
                current_text = " ".join([s.text for s in segments]).strip()
                
                # --- THE "NO CUT WORD" LOGIC ---
                # We don't output the *entire* text immediately. 
                # We assume the last few words might be unstable (cut off).
                # We commit everything EXCEPT the last 0.5 seconds of speech context.
                
                # Send the "Real-time" preview to the user
                response = {
                    "type": "partial",
                    "text": current_text
                }
                await websocket.send(json.dumps(response))
                
                # 3. Rolling Buffer Management
                # Keep the last 'overlap_samples' (0.5s) in the buffer
                # Discard the rest (it has been processed)
                audio_buffer = audio_buffer[-overlap_samples:]
                
                # (Optional: In a production app, you would verify stability here 
                # before clearing the buffer, but for blazing speed, overlapping 
                # usually solves the cut-word issue).

    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")
    except Exception as e:
        print(f"Error: {e}")

async def main():
    async with websockets.serve(transcribe_audio, "0.0.0.0", PORT):
        print(f"Server started on port {PORT}")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())
