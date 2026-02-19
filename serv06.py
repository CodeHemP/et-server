import asyncio
import websockets
import json
import numpy as np
import torch
import os
import ctranslate2
from faster_whisper import WhisperModel
from transformers import NllbTokenizerFast

# --- CONFIGURATION ---
PORT = 8765
WHISPER_MODEL_ID = "deepdml/faster-whisper-large-v3-turbo-ct2"
OFFICIAL_NLLB_MODEL = "facebook/nllb-200-distilled-600M"
LOCAL_NLLB_DIR = "nllb-200-ct2-int8"

# Language Codes for NLLB-200
TARGET_LANGS = {
    "HI": "hin_Deva", # Hindi
    "VI": "vie_Latn", # Vietnamese
    "DZ": "dzo_Tibt"  # Bhutanese (Dzongkha)
}

print("--------------------------------------------------")
print("1. Checking/Converting Translation Model...")

if not os.path.exists(LOCAL_NLLB_DIR):
    print(f"   Downloading and converting {OFFICIAL_NLLB_MODEL}...")
    # This command is more stable for M2M100/NLLB models
    os.system(f"ct2-transformers-converter --model {OFFICIAL_NLLB_MODEL} --output_dir {LOCAL_NLLB_DIR} --quantization int8 --force")
    
if not os.path.exists(LOCAL_NLLB_DIR):
    print("❌ Conversion failed again. Trying fallback method...")
    # Fallback: using the Python API directly with specific settings
    import ctranslate2.converters
    converter = ctranslate2.converters.TransformersConverter(OFFICIAL_NLLB_MODEL)
    converter.convert(LOCAL_NLLB_DIR, quantization="int8", force=True)

# LOAD MODELS
print("   Loading Models into GPU...")
translator = ctranslate2.Translator(LOCAL_NLLB_DIR, device="cuda")
tokenizer = NllbTokenizerFast.from_pretrained(OFFICIAL_NLLB_MODEL)

print("2. Loading Whisper (ASR)...")
asr_model = WhisperModel(WHISPER_MODEL_ID, device="cuda", compute_type="int8_float16")

print("3. Loading VAD...")
vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', onnx=True)
(get_speech_timestamps, _, _, _, _) = utils

print("--------------------------------------------------")
print(f"✅ BLAZING FAST SYSTEM READY on Port {PORT}")

# --- AI LOGIC ---

def translate_phrase(text, target_lang):
    # Tokenize for NLLB
    tokenizer.src_lang = "eng_Latn"
    source = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
    # Translate
    results = translator.translate_batch([source], target_prefix=[[target_lang]])
    target_tokens = results[0].hypotheses[0]
    # Decode
    return tokenizer.decode(tokenizer.convert_tokens_to_ids(target_tokens), skip_special_tokens=True)

async def handle_client(websocket):
    print("--> Client Connected")
    audio_buffer = np.array([], dtype=np.float32)
    
    try:
        async for message in websocket:
            # 1. Collect Audio
            chunk = np.frombuffer(message, dtype=np.float32)
            audio_buffer = np.concatenate((audio_buffer, chunk))
            
            # 2. Process every 2.5 seconds
            if len(audio_buffer) >= 40000: # ~2.5s @ 16kHz
                
                # VAD Check (Speech detection)
                recent_audio = torch.tensor(audio_buffer[-16000:])
                if vad_model(recent_audio, 16000).item() < 0.4:
                    audio_buffer = audio_buffer[-8000:] # Keep small tail
                    continue

                # 3. Transcribe English
                segments, _ = asr_model.transcribe(audio_buffer, beam_size=1, language="en")
                text_en = " ".join([s.text.strip() for s in segments])

                if len(text_en) > 3:
                    # 4. Translate to all 3 languages
                    res_hi = translate_phrase(text_en, TARGET_LANGS["HI"])
                    res_vi = translate_phrase(text_en, TARGET_LANGS["VI"])
                    res_dz = translate_phrase(text_en, TARGET_LANGS["DZ"])

                    # 5. Send results
                    payload = {
                        "EN": text_en,
                        "HI": res_hi,
                        "VI": res_vi,
                        "DZ": res_dz
                    }
                    await websocket.send(json.dumps(payload))
                
                # Rolling buffer
                audio_buffer = audio_buffer[-16000:]

    except Exception as e:
        print(f"Connection Error: {e}")
    finally:
        print("--> Client Disconnected")

async def main():
    async with websockets.serve(handle_client, "0.0.0.0", PORT):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
