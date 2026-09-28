from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import ctranslate2
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
from pathlib import Path
import re
import os

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
translator = ctranslate2.Translator(str(model_path / "ct2-int8_float32"), device="cpu", compute_type="int8_float32")
print("Mô hình đã sẵn sàng!")

def clean_vietnamese(text: str) -> str:
    # Thêm khoảng trắng sau dấu câu nếu thiếu
    text = re.sub(r'([.!?])([A-ZÀ-Ỹ0-9])', r'\1 \2', text)
    text = re.sub(r'([,;:])([a-zA-Zà-ỹÀ-Ỹ0-9])', r'\1 \2', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()

def translate_text(text: str) -> str:
    if not text or not text.strip():
        return ""
    
    # Kiểm tra nếu không có tiếng Trung thì không dịch để tránh lỗi mô hình Marian
    has_chinese = bool(re.search(r'[\u4e00-\u9fa5]', text))
    if not has_chinese:
        return text.strip()

    paragraphs = text.split("\n")
    results = []
    
    for para in paragraphs:
        para_clean = para.strip()
        if not para_clean:
            results.append("")
            continue
        
        # Nếu đoạn không có chữ Hán thì giữ nguyên
        if not re.search(r'[\u4e00-\u9fa5]', para_clean):
            results.append(para_clean)
            continue
        
        sentences = [s for s in re.split(r'([。！？\n])', para_clean) if s]
        chunks = []
        cur = ""
        for s in sentences:
            cur += s
            if s in ['。', '！', '？', '\n'] or len(cur) > 80:
                chunks.append(cur)
                cur = ""
        if cur:
            chunks.append(cur)
        
        tokens = [tokenizer.convert_ids_to_tokens(tokenizer(c).input_ids) for c in chunks]
        res = translator.translate_batch(tokens, beam_size=1, max_decoding_length=256)
        out = "".join(tokenizer.decode(tokenizer.convert_tokens_to_ids(r.hypotheses[0]), skip_special_tokens=True) for r in res)
        results.append(clean_vietnamese(out))
        
    return "\n\n".join(results)

@app.get("/")
def home():
    return {"status": "ok", "model": "HachimiMT-60-QT"}

@app.post("/translate")
async def api_translate(req: Request):
    data = await req.json()
    raw_text = data.get("text", "")
    translated = translate_text(raw_text)
    return JSONResponse({"result": translated})
