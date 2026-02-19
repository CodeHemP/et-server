import asyncio
import websockets
import json
import numpy as np
import torch
import os
import subprocess
from faster_whisper import WhisperModel
import ctranslate2
from transformers import NllbTokenizerFast

# --- CONFIGURATION ---
PORT = 8765
WHISPER_MODEL_ID = "deepdml/faster-whisper-large-v3-turbo-ct2"
# We will download the official model and convert it locally
OFFICIAL_NLLB_MODEL = "facebook/nllb-200-distilled-600M"
LOCAL_NLLB_DIR = "nllb-200-ct2-int8"

# Language Codes
TARGET_LANGS = {
    "HI": "hin_Deva", # Hindi
    "VI": "vie_Latn", # Vietnamese
    "DZ": "dzo_Tibt"  # Bhutanese (Dzongkha)
}

print("--------------------------------------------------")
print("1. Checking Translation Model...")

# AUTO-CONVERTER: If we haven't converted the model yet, do it now.
if not os.path.exists(LOCAL_NLLB_DIR):
    print(f"   Converting {OFFICIAL_NLLB_MODEL} to CTranslate2 INT8...")
    print("   (This takes ~60 seconds only once)")
    
    # Run the conversion command
    command = [
        "ct2-transformers-converter",
        "--model", OFFICIAL_NLLB_MODEL,
        "--output_dir", LOCAL_NLLB_DIR,
        "--quantization", "int8",
        "--force"
    ]
    try:
        subprocess.run(command, check=True)
        print("   ✅ Conversion Complete!")
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Conversion Failed: {e}")
        exit(1)
else:
    print("   ✅ Model already converted and ready.")

# Load the locally converted model
translator = ctranslate2.Translator(LOCAL_NLLB_DIR, device="cuda", compute_type="int8")
tokenizer = NllbTokenizerFast.from_pretrained(OFFICIAL_NLLB_MODEL, src_lang="eng_Latn")
print("   NLLB Loaded.")

print("2. Loading Whisper (ASR)...")
asr_model = WhisperModel(WHISPER_MODEL_ID, device="cuda", compute_type="int8_float16")
print("   Whisper Loaded.")

print("3. Loading VAD...")
vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', onnx=True)
(get_speech_timestamps, _, _, _, _) = utils
print("   VAD Loaded.")
print("--------------------------------------------------")
print(f"✅ SERVER READY on Port {PORT}")

# --- HELPER FUNCTIONS ---

def is_speech(audio_chunk):
    """Returns True if audio contains speech (VAD)"""
    tensor = torch.tensor(audio_chunk)
    prob = vad_model(tensor, 16000).item()
    return prob > 0.5

def translate_batch(text, target_langs_dict):
    """Translates text into multiple languages"""
    if not text or len(text.strip()) < 2: 
        return {code: "" for code in target_langs_dict}
    
    source = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
    results = {}
    
    for code, lang_id in target_langs_dict.items():
        try:
            # Batch size 1 is fastest for real-time
            translation_result = translator.translate_batch(
                [source], 
                target_prefix=[[lang_id]]
            )
            target_tokens = translation_result[0].hypotheses[0]
            decoded = tokenizer.decode(tokenizer.convert_tokens_to_ids(target_tokens))
            results[code] = decoded
        except Exception as e:
            results[code] = "Error"
            
    return results

async def transcribe_handler(websocket):
    print("--> Client Connected")
    
    audio_buffer = np.array([], dtype=np.float32)
    SAMPLE_RATE = 16000
    PROCESS_INTERVAL = 2.5 
    
    try:
        async for message in websocket:
            chunk = np.frombuffer(message, dtype=np.float32)
            audio_buffer = np.concatenate((audio_buffer, chunk))
            
            if len(audio_buffer) >= int(SAMPLE_RATE * PROCESS_INTERVAL):
                
                # VAD Check (Last 1.0s)
                if not is_speech(audio_buffer[-16000:]):
                    audio_buffer = audio_buffer[-8000:] 
                    continue
                
                # Transcribe
                segments, _ = asr_model.transcribe(
                    audio_buffer, 
                    beam_size=1, 
                    language="en",
                    condition_on_previous_text=False,
                    vad_filter=True,
                    initial_prompt="Indian English, technical terms."
                )
                
                text_en = " ".join([s.text.strip() for s in segments if s.no_speech_prob < 0.6])
                
                if text_en and len(text_en) > 2:
                    # Translate
                    translations = translate_batch(text_en, TARGET_LANGS)
                    
                    response = {
                        "EN": text_en,
                        "HI": translations["HI"],
                        "VI": translations["VI"],
                        "DZ": translations["DZ"]
                    }
                    await websocket.send(json.dumps(response))
                
                # Buffer Management
                audio_buffer = audio_buffer[-16000:]
                
    except Exception as e:
        print(f"Server Error: {e}")
    finally:
        print("--> Client Disconnected")

async def main():
    async with websockets.serve(transcribe_handler, "0.0.0.0", PORT):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
