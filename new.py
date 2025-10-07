import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model

# -------- Step 1: Ask user for dataset --------
file_path = input("📂 Enter path to JSON dataset (press Enter to use fallback): ").strip()

if file_path == "":
    print("⚠️ No file given, using fallback dataset.")
    raw_data = [
        {
            "instruction": "What is AI?",
            "input": "",
            "output": "AI is the simulation of human intelligence by machines."
        },
        {
            "instruction": "Why use AI?",
            "input": "",
            "output": "AI helps automate tasks and make predictions."
        }
    ]
else:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        print(f"✅ Loaded dataset with {len(raw_data)} examples.")
    except Exception as e:
        print("❌ Error loading dataset:", e)
        print("⚠️ Falling back to default dataset.")
        raw_data = [
            {
                "instruction": "What is AI?",
                "input": "",
                "output": "AI is the simulation of human intelligence by machines."
            }
        ]

# -------- Step 2: Convert dataset --------
train_data = []
for entry in raw_data:
    instruction = entry.get("instruction", "")
    context = entry.get("input", "")
    output = entry.get("output", "")

    if context:
        prompt = f"Instruction: {instruction}\nInput: {context}\nAnswer:"
    else:
        prompt = f"Instruction: {instruction}\nAnswer:"

    train_data.append({"prompt": prompt, "response": output})

dataset = Dataset.from_list(train_data)

# -------- Step 3: Load TinyLLaMA --------
MODEL_NAME = "TinyLLaMA/TinyLLaMA-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,   # CPU safe
    device_map="cpu"
)

# -------- Step 4: Tokenize --------
def tokenize_function(example):
    text = example["prompt"] + " " + example["response"]
    tokens = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=256
    )
    # Add labels (same as input_ids for causal LM)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


tokenized_dataset = dataset.map(tokenize_function, batched=False)

# -------- Step 5: Apply LoRA --------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# -------- Step 6: Training setup --------
training_args = TrainingArguments(
    output_dir="./tinyllama-finetuned",
    per_device_train_batch_size=1,  # CPU safe
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    remove_unused_columns=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# -------- Step 7: Train --------
print("🚀 Starting training...")
trainer.train()

# -------- Step 8: Save model --------
model.save_pretrained("./tinyllama-finetuned")
tokenizer.save_pretrained("./tinyllama-finetuned")

# -------- Step 9: Inference --------
def chat(question):
    prompt = f"Instruction: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n🤖 Chatbot ready! Type your question.")
while True:
    q = input("You: ")
    if q.lower() in ["quit", "exit"]:
        break
    print("Bot:", chat(q))
