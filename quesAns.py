import json
import os
from transformers import pipeline
from tqdm import tqdm

# ============ Configuration ============
INPUT_FILE = r"C:\Users\Maha Rehan\Documents\Ai-for-Org\outputs\PM4DEV_Project_Scope_Management_cleaned.txt"
OUTPUT_FILE = os.path.splitext(INPUT_FILE)[0] + "_QA.json"

# Load a pre-trained Q&A generation model
print("🤖 Loading model for question generation...")
qa_generator = pipeline("text2text-generation", model="valhalla/t5-small-qg-prepend")
print("✅ Model loaded.\n")

# ============ Helper Function ============
def chunk_text(text, max_words=150):
    """Split long text into smaller passages."""
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

# ============ Main Function ============
def generate_qa_from_text(text):
    qas = []
    chunks = chunk_text(text)
    for chunk in tqdm(chunks, desc="🧠 Generating Questions"):
        try:
            result = qa_generator(chunk, max_length=128, do_sample=True, temperature=0.9)
            generated = result[0]["generated_text"].strip()
            if "?" in generated:
                question, *answer_parts = generated.split("?")
                question = question.strip() + "?"
                answer = " ".join(answer_parts).strip()
                qas.append({
                    "question": question,
                    "answer": answer if answer else "Not explicitly answered.",
                    "supporting_passages": [chunk]
                })
        except Exception as e:
            print("⚠️ Error generating Q&A for a chunk:", e)
    return qas

# ============ Main CLI Run ============
if __name__ == "__main__":
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"📘 Generating Q&A pairs from: {os.path.basename(INPUT_FILE)}")
    qa_pairs = generate_qa_from_text(text)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, indent=4, ensure_ascii=False)

    print(f"\n✅ Saved {len(qa_pairs)} Q&A pairs → {OUTPUT_FILE}")
