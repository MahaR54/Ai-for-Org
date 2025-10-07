import fitz
import re
import json
import os
import pytesseract
from PIL import Image
from datasketch import MinHash, MinHashLSH
from trafilatura import extract
import docx2txt
from hashlib import sha256
from tqdm import tqdm
from datetime import datetime
from langdetect import detect, DetectorFactory

# ============ Configuration ============
DetectorFactory.seed = 0  # ensure consistent language detection
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("🔤 Using lightweight language detector (langdetect)... ✅\n")

# ============ Constants ============
UNWANTED_PATTERNS = [
    'subscribe', 'follow us', 'click here', 'share on', 'cookie policy',
    'advertisement', 'back to top', 'comments?', 'login', 'sign up', 'terms of service'
]
FILTER_LOGS = []
seen_spans = set()
lsh_index = MinHashLSH(threshold=0.8, num_perm=128)

# ============ Core Functions ============

def extract_text_from_pdf(path):
    try:
        with fitz.open(path) as doc:
            extracted_pages = [page.get_text("text") for page in doc]
            full_text = "\n".join(extracted_pages)
            return full_text if full_text.strip() else ""
    except Exception as e:
        print(f"❌ PDF extraction failed: {e}")
        return ""

def extract_text_with_ocr(path):
    print(f"🧠 OCR extraction for {os.path.basename(path)}")
    try:
        doc = fitz.open(path)
        text = ""
        for page in tqdm(doc, desc="OCR Progress"):
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text += pytesseract.image_to_string(img) + "\n"
        return text
    except Exception as e:
        print(f"⚠️ OCR failed for {path}: {e}")
        return ""

def extract_main_text(html):
    return extract(html, include_comments=False, include_tables=False)

def extract_text_from_docx(path):
    try:
        return docx2txt.process(path)
    except Exception as e:
        print(f"⚠️ DOCX extraction failed: {e}")
        return ""

def is_english(text):
    try:
        lang = detect(text.replace('\n', ' '))
        return lang == 'en'
    except:
        return False

def remove_bad_lines(text):
    lines = text.split('\n')
    cleaned = [line for line in lines if not any(p in line.lower() for p in UNWANTED_PATTERNS)]
    return '\n'.join(cleaned) if len(cleaned) >= 0.95 * len(lines) else None

def has_exact_duplicate_spans(text, seen_spans, min_tokens=50):
    tokens = text.split()
    for i in range(len(tokens) - min_tokens + 1):
        span = ' '.join(tokens[i:i + min_tokens])
        span_hash = sha256(span.encode()).hexdigest()
        if span_hash in seen_spans:
            return True
        seen_spans.add(span_hash)
    return False

def get_minhash(text, num_perm=128):
    m = MinHash(num_perm=num_perm)
    for word in set(text.split()):
        m.update(word.encode('utf8'))
    return m

def clean_text(text, skip_filters=False):
    if not text:
        FILTER_LOGS.append("🚫 Skipped: empty text")
        return None
    if not skip_filters and not is_english(text):
        FILTER_LOGS.append("🚫 Skipped: not English")
        return None
    if not skip_filters and len(text.split()) < 50:
        FILTER_LOGS.append("🚫 Skipped: too short (<50 words)")
        return None
    if not skip_filters:
        text = remove_bad_lines(text)
        if text is None:
            FILTER_LOGS.append("🚫 Skipped: filtered by bad lines")
            return None
        if has_exact_duplicate_spans(text, seen_spans):
            FILTER_LOGS.append("🚫 Skipped: duplicate span")
            return None
        m = get_minhash(text)
        if lsh_index.query(m):
            FILTER_LOGS.append("🚫 Skipped: duplicate LSH")
            return None
        lsh_index.insert(str(hash(text)), m)
    return text

def chunk_text(text, max_tokens=512):
    words = text.split()
    return [" ".join(words[i:i + max_tokens]) for i in range(0, len(words), max_tokens)]

# ============ Main CLI Runner ============

def process_file(path, skip_filters=False):
    ext = os.path.splitext(path)[1].lower()
    text = ""
    if ext == ".pdf":
        text = extract_text_from_pdf(path)
        if not text.strip():
            text = extract_text_with_ocr(path)
    elif ext == ".docx":
        text = extract_text_from_docx(path)
    elif ext == ".html":
        with open(path, "r", encoding="utf-8") as f:
            html = f.read()
        text = extract_main_text(html)
    elif ext == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        print(f"⚠️ Unsupported file type: {ext}")
        return

    if not text:
        print(f"🚫 No text extracted from {path}")
        return

    cleaned = clean_text(text, skip_filters=skip_filters)
    if not cleaned:
        print(f"🚫 Skipped {path} (did not pass filters).")
        return

    chunks = chunk_text(cleaned)
    metadata = {
        "filename": os.path.basename(path),
        "language": "en",
        "word_count": len(cleaned.split()),
        "num_chunks": len(chunks),
        "processed_at": datetime.now().isoformat()
    }

    # Save outputs
    base = os.path.splitext(os.path.basename(path))[0]
    with open(os.path.join(OUTPUT_DIR, f"{base}_cleaned.txt"), "w", encoding="utf-8") as f:
        f.write(cleaned)
    with open(os.path.join(OUTPUT_DIR, f"{base}_chunks.jsonl"), "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            f.write(json.dumps({"chunk_id": i, "content": chunk}) + "\n")
    with open(os.path.join(OUTPUT_DIR, f"{base}_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Processed {path} → {base}_cleaned.txt, {base}_chunks.jsonl, {base}_metadata.json")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MDR 7-Stage Text Cleaner (CLI Version)")
    parser.add_argument("input", help="Path to file or folder to process")
    parser.add_argument("--skip-filters", action="store_true", help="Skip filtering rules (debug mode)")
    args = parser.parse_args()

    input_path = args.input
    if os.path.isdir(input_path):
        files = [os.path.join(input_path, f) for f in os.listdir(input_path)
                 if f.lower().endswith((".pdf", ".docx", ".html", ".txt"))]
    else:
        files = [input_path]

    for fpath in files:
        process_file(fpath, skip_filters=args.skip_filters)

    if FILTER_LOGS:
        print("\n🧾 Filter Logs:")
        for log in FILTER_LOGS:
            print(" -", log)
