from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import ctranslate2
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
from pathlib import Path
import re
import os
import uvicorn
import asyncio
from concurrent.futures import ThreadPoolExecutor

app = FastAPI(title="HachimiMT API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Đang nạp mô hình HachimiMT-60-QT CTranslate2...")
repo = "ngocdang83/HachimiMT-60-QT"
model_path = Path(snapshot_download(repo, allow_patterns=[
    "config.json", "source.spm", "target.spm", "vocab.json",
    "tokenizer_config.json", "ct2-int8_float32/*"
]))

tokenizer = AutoTokenizer.from_pretrained(str(model_path))
translator = ctranslate2.Translator(
    str(model_path / "ct2-int8_float32"), 
    device="cpu", 
    compute_type="int8_float32",
    intra_threads=2,
    inter_threads=1
)
print("Mô hình đã sẵn sàng!")

executor = ThreadPoolExecutor(max_workers=2)

def clean_vietnamese(text: str) -> str:
    text = re.sub(r'([.!?])([A-ZÀ-Ỹ0-9])', r'\1 \2', text)
    text = re.sub(r'([,;:])([a-zA-Zà-ỹÀ-Ỹ0-9])', r'\1 \2', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()

def translate_text(text: str) -> str:
    if not text or not text.strip():
        return ""
    
    if not re.search(r'[\u4e00-\u9fa5]', text):
        return text.strip()

    raw_paragraphs = [p.strip() for p in text.split("\n")]
    valid_indices = []
    tokens_batch = []
    
    for idx, para in enumerate(raw_paragraphs):
        if para and re.search(r'[\u4e00-\u9fa5]', para):
            valid_indices.append(idx)
            tokens_batch.append(tokenizer.convert_ids_to_tokens(tokenizer(para).input_ids))
            
    if not tokens_batch:
        return text.strip()
        
    # Translate entire batch in one single CTranslate2 C++ call!
    res = translator.translate_batch(tokens_batch, beam_size=1, max_decoding_length=256)
    
    translated_paras = list(raw_paragraphs)
    for i, orig_idx in enumerate(valid_indices):
        out = tokenizer.decode(tokenizer.convert_tokens_to_ids(res[i].hypotheses[0]), skip_special_tokens=True)
        translated_paras[orig_idx] = clean_vietnamese(out)
        
    # Filter empty and join
    return "\n\n".join([p for p in translated_paras if p])

@app.get("/")
def home():
    return {"status": "ok", "model": "HachimiMT-60-QT"}

@app.post("/translate")
async def api_translate(req: Request):
    data = await req.json()
    raw_text = data.get("text", "")
    
    # Run CPU translation in ThreadPool so asyncio event loop never blocks!
    loop = asyncio.get_event_loop()
    translated = await loop.run_in_executor(executor, translate_text, raw_text)
    
    return JSONResponse({"result": translated})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"Khởi động Uvicorn server tại cổng {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
