import asyncio
import websockets
import json
import numpy as np
import torch
import os
from faster_whisper import WhisperModel
import ctranslate2
from transformers import NllbTokenizerFast

# --- CONFIGURATION ---
PORT = 8765
WHISPER_MODEL = "deepdml/faster-whisper-large-v3-turbo-ct2"
# We use a quantized version of NLLB for extreme speed
TRANSLATION_MODEL = "Softcatala/nllb-200-distilled-600M-ct2-int8"

# Language Codes for NLLB
SRC_LANG = "eng_Latn" # English
TARGET_LANGS = {
    "HI": "hin_Deva", # Hindi
    "VI": "vie_Latn", # Vietnamese
    "DZ": "dzo_Tibt"  # Bhutanese (Dzongkha)
}

print("1. Loading VAD...")
vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', onnx=True)
(get_speech_timestamps, _, _, _, _) = utils

print("2. Loading Whisper (ASR)...")
asr_model = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="int8_float16")

print("3. Loading Translator (NLLB)...")
# Download/Load the tokenizer
tokenizer = NllbTokenizerFast.from_pretrained("facebook/nllb-200-distilled-600M", src_lang=SRC_LANG)
# Load the optimized CTranslate2 translation engine
translator = ctranslate2.Translator(TRANSLATION_MODEL, device="cuda", compute_type="int8")

print(f"✅ System Ready on Port {PORT}")

# Helper: Detect Speech
def is_speech(audio_chunk):
    tensor = torch.tensor(audio_chunk)
    prob = vad_model(tensor, 16000).item()
    return prob > 0.6

# Helper: Translate Text
def translate_text(text, target_lang_code):
    if not text or len(text) < 2: return ""
    
    # 1. Tokenize
    source = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
    
    # 2. Translate (Beam size 1 for speed)
    results = translator.translate_batch([source], target_prefix=[[target_lang_code]])
    target = results[0].hypotheses[0]
    
    # 3. Decode
    return tokenizer.decode(tokenizer.convert_tokens_to_ids(target))

async def transcribe_handler(websocket):
    print("--> User Connected")
    audio_buffer = np.array([], dtype=np.float32)
    SAMPLE_RATE = 16000
    PROCESS_INTERVAL = 2.5 # Process every 2.5s for better context
    
    try:
        async for message in websocket:
            chunk = np.frombuffer(message, dtype=np.float32)
            audio_buffer = np.concatenate((audio_buffer, chunk))
            
            if len(audio_buffer) >= int(SAMPLE_RATE * PROCESS_INTERVAL):
                
                # VAD Check (Last 1.0s)
                if not is_speech(audio_buffer[-16000:]):
                    # Silence -> Trim buffer and skip
                    audio_buffer = audio_buffer[-8000:] 
                    continue
                
                # 1. Transcribe (English)
                segments, _ = asr_model.transcribe(
                    audio_buffer, 
                    beam_size=1, 
                    language="en",
                    condition_on_previous_text=False,
                    vad_filter=True
                )
                
                text_en = " ".join([s.text.strip() for s in segments if s.no_speech_prob < 0.5])
                
                if text_en:
                    # 2. Translate (Parallel)
                    # We run translations sequentially here, but it takes <10ms on a 4090
                    translations = {}
                    for code, lang_id in TARGET_LANGS.items():
                        translations[code] = translate_text(text_en, lang_id)
                    
                    # 3. Send Bundle
                    response = {
                        "EN": text_en,
                        "HI": translations["HI"],
                        "VI": translations["VI"],
                        "DZ": translations["DZ"]
                    }
                    await websocket.send(json.dumps(response))
                
                # Keep last 1s context
                audio_buffer = audio_buffer[-16000:]
                
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("--> User Disconnected")

async def main():
    async with websockets.serve(transcribe_handler, "0.0.0.0", PORT):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
