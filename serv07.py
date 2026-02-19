import asyncio
import websockets
import json
import numpy as np
import torch
from faster_whisper import WhisperModel

# --- CONFIGURATION ---
PORT = 8765
MODEL_SIZE = "deepdml/faster-whisper-large-v3-turbo-ct2"

# 1. LOAD SILERO VAD (The Gatekeeper)
print("Loading VAD...")
vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model='silero_vad',
                                  force_reload=False,
                                  onnx=True)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
print("VAD Loaded.")

# 2. LOAD WHISPER
print("Loading Whisper...")
model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="int8_float16")
print("System Ready.")

# --- NEW STRATEGY: KEYWORD PROMPT ---
# Instead of a sentence (which causes loops), we use keywords.
# This tells the model the "vibe" without giving it a sentence to repeat.
INITIAL_PROMPT = "Indian English, Technical, Hindi terms, clear speech, accurate transcript."

def check_speech_confidence(audio_chunk, sr=16000):
    """
    Returns True if the chunk contains human speech.
    """
    # Convert numpy float32 to tensor
    tensor_audio = torch.tensor(audio_chunk)
    
    # Get probability of speech (0.0 to 1.0)
    speech_prob = vad_model(tensor_audio, sr).item()
    
    # STRICT THRESHOLD: Only process if > 80% confident it's speech
    return speech_prob > 0.8

async def transcribe_audio(websocket):
    print("--> Client connected")
    
    # Buffer config
    audio_buffer = np.array([], dtype=np.float32)
    SAMPLE_RATE = 16000
    # Process every 2.0 seconds
    PROCESS_INTERVAL = 2.0 
    chunk_samples = int(SAMPLE_RATE * PROCESS_INTERVAL)
    
    try:
        async for message in websocket:
            # Receive audio
            chunk = np.frombuffer(message, dtype=np.float32)
            audio_buffer = np.concatenate((audio_buffer, chunk))
            
            # Check if we have enough data to process
            if len(audio_buffer) >= chunk_samples:
                
                # --- THE HARD GATE ---
                # Check the LAST 1.5 seconds of audio for speech activity.
                # If it's just static/silence, DROP IT.
                validation_window = audio_buffer[-int(SAMPLE_RATE*1.5):]
                
                if not check_speech_confidence(validation_window):
                    # NO SPEECH DETECTED -> Discard old buffer to prevent buildup
                    # Keep only a tiny tail (0.5s) for continuity
                    audio_buffer = audio_buffer[-int(SAMPLE_RATE*0.5):]
                    
                    # Tell client to clear line or stay silent (optional)
                    continue 

                # SPEECH DETECTED -> Transcribe
                segments, info = model.transcribe(
                    audio_buffer, 
                    beam_size=1,            # Faster, less creative
                    temperature=0.0,        # STRICT: No creativity (prevents hallucination)
                    condition_on_previous_text=False, # KEY FIX: Don't look at past errors
                    initial_prompt=INITIAL_PROMPT,
                    vad_filter=True,        # Double filtering (internal Whisper VAD)
                    vad_parameters=dict(min_silence_duration_ms=500)
                )
                
                text_segments = []
                for s in segments:
                    # Filter out short garbage (e.g. "Thank you", "Bye") common in hallucinations
                    if s.no_speech_prob < 0.6: 
                        text_segments.append(s.text.strip())

                final_text = " ".join(text_segments)

                if final_text:
                    # Send result
                    response = {"text": final_text}
                    await websocket.send(json.dumps(response))
                
                # Buffer Management:
                # Keep last 1.0s so we don't cut words between chunks
                audio_buffer = audio_buffer[-int(SAMPLE_RATE*1.0):]

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
