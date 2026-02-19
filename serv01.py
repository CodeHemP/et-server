import asyncio
import websockets
import json
import numpy as np
import torch
from faster_whisper import WhisperModel

# --- CONFIGURATION ---
PORT = 8765
# Use a slightly more robust model config
MODEL_SIZE = "deepdml/faster-whisper-large-v3-turbo-ct2"

print("1. Loading VAD Model...")
# Load Silero VAD (Fast and accurate speech detection)
vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model='silero_vad',
                                  force_reload=False,
                                  onnx=True) # ONNX is faster
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
print("VAD Loaded.")

print("2. Loading Whisper Model...")
# Run on GPU (cuda) with INT8 quantization for speed
model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="int8_float16")
print("Whisper Loaded! Ready for connections.")

# Prompt to guide the style
INITIAL_PROMPT = "The following is a transcript of an Indian English speaker. The output is strictly in English."

def is_speech(audio_chunk, sr=16000):
    # Silero expects a Tensor
    audio_tensor = torch.tensor(audio_chunk)
    # Get speech probability (0.0 to 1.0)
    speech_prob = vad_model(audio_tensor, sr).item()
    return speech_prob > 0.5

async def transcribe_audio(websocket):
    print("--> Client connected")
    
    audio_buffer = np.array([], dtype=np.float32)
    committed_text = ""
    
    # 2.5 seconds window gives better context than 2.0
    SAMPLE_RATE = 16000
    CHUNK_DURATION = 2.5 
    chunk_samples = int(SAMPLE_RATE * CHUNK_DURATION)
    
    try:
        async for message in websocket:
            chunk = np.frombuffer(message, dtype=np.float32)
            audio_buffer = np.concatenate((audio_buffer, chunk))
            
            # Only process if buffer is full enough
            if len(audio_buffer) >= chunk_samples:
                
                # --- CRITICAL FIX: VAD CHECK ---
                # Check the last 1 second for speech. If it's silence, don't hallucinate.
                check_window = audio_buffer[-16000:] # Check last 1 second
                if not is_speech(check_window):
                    # It's silence. Just clear a bit of buffer and wait.
                    # We keep the buffer overlapping but don't run heavy inference
                    # This prevents "The following is a transcript..." loops.
                    audio_buffer = audio_buffer[-8000:] # Keep last 0.5s for continuity
                    continue 

                # If speech is detected, transcribe
                segments, info = model.transcribe(
                    audio_buffer, 
                    beam_size=5, 
                    language="en",
                    condition_on_previous_text=False, # Helps prevent loops
                    initial_prompt=INITIAL_PROMPT,
                    vad_filter=True, # Built-in filter as a backup
                    vad_parameters=dict(min_silence_duration_ms=500)
                )
                
                text_segments = [s.text.strip() for s in segments]
                current_text = " ".join(text_segments)
                
                # Send back to client
                if current_text:
                    response = {
                        "text": current_text,
                        "is_final": False # Streaming mode
                    }
                    await websocket.send(json.dumps(response))
                
                # Rolling buffer: Keep last 0.5s (8000 samples) to prevent cut words
                audio_buffer = audio_buffer[-8000:]

    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("--> Client disconnected")

async def main():
    async with websockets.serve(transcribe_audio, "0.0.0.0", PORT):
        print(f"Server started on port {PORT}")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
