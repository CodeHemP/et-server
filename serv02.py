import asyncio
import websockets
import json
import numpy as np
import torch
import os
from faster_whisper import WhisperModel
import ctranslate2
from transformers import NllbTokenizerFast
from huggingface_hub import snapshot_download

# --- CONFIGURATION ---
PORT = 8765
WHISPER_MODEL_ID = "deepdml/faster-whisper-large-v3-turbo-ct2"
NLLB_MODEL_ID = "Softcatala/nllb-200-distilled-600M-ct2-int8"

# Language Codes
TARGET_LANGS = {
    "HI": "hin_Deva", # Hindi
    "VI": "vie_Latn", # Vietnamese
    "DZ": "dzo_Tibt"  # Bhutanese (Dzongkha)
}

print("--------------------------------------------------")
print("1. Downloading/Loading Translation Model (NLLB)...")
# Download the model explicitly to ensure path exists
model_path = snapshot_download(repo_id=NLLB_MODEL_ID)
translator = ctranslate2.Translator(model_path, device="cuda", compute_type="int8")
tokenizer = NllbTokenizerFast.from_pretrained("facebook/nllb-200-distilled-600M", src_lang="eng_Latn")
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
    """Translates text into multiple languages efficiently"""
    if not text or len(text.strip()) < 2: 
        return {code: "" for code in target_langs_dict}
    
    # 1. Tokenize Source
    source = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
    
    # 2. Prepare Batch
    results = {}
    
    # We do this in a loop because one-to-many is tricky in pure batch, 
    # but CTranslate2 is so fast on 4090 it doesn't matter for 3 langs.
    for code, lang_id in target_langs_dict.items():
        try:
            translation_result = translator.translate_batch(
                [source], 
                target_prefix=[[lang_id]]
            )
            target_tokens = translation_result[0].hypotheses[0]
            decoded = tokenizer.decode(tokenizer.convert_tokens_to_ids(target_tokens))
            results[code] = decoded
        except Exception as e:
            print(f"Translation Error ({code}): {e}")
            results[code] = "Error"
            
    return results

async def transcribe_handler(websocket):
    print("--> Client Connected")
    
    audio_buffer = np.array([], dtype=np.float32)
    SAMPLE_RATE = 16000
    PROCESS_INTERVAL = 2.5 # Process every 2.5s
    
    try:
        async for message in websocket:
            chunk = np.frombuffer(message, dtype=np.float32)
            audio_buffer = np.concatenate((audio_buffer, chunk))
            
            # Process when buffer is full enough
            if len(audio_buffer) >= int(SAMPLE_RATE * PROCESS_INTERVAL):
                
                # 1. VAD Check (Check last 1.0s)
                # If it's silence, skip inference to save compute/prevent hallucinations
                if not is_speech(audio_buffer[-16000:]):
                    # Keep overlap and continue
                    audio_buffer = audio_buffer[-8000:] 
                    continue
                
                # 2. Transcribe (English)
                segments, _ = asr_model.transcribe(
                    audio_buffer, 
                    beam_size=1, 
                    language="en",
                    condition_on_previous_text=False, # Prevents loops
                    vad_filter=True,
                    initial_prompt="Indian English, clear speech."
                )
                
                # Collect valid segments
                text_en = " ".join([s.text.strip() for s in segments if s.no_speech_prob < 0.6])
                
                if text_en and len(text_en) > 2:
                    # 3. Translate
                    translations = translate_batch(text_en, TARGET_LANGS)
                    
                    # 4. Construct Response (The Keys the client expects)
                    response = {
                        "EN": text_en,
                        "HI": translations["HI"],
                        "VI": translations["VI"],
                        "DZ": translations["DZ"]
                    }
                    
                    # 5. Send
                    await websocket.send(json.dumps(response))
                
                # Buffer Management: Keep last 1s context
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
