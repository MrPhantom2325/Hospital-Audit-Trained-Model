import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "PhantomAjusshi/phi3-auditor-merged"

print("Loading model and tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
)

print("Model loaded successfully.")

def generate_response(report_text: str, instruction: str = "Analyze the clinical model report and classify its health."):
    prompt = (
        f"<|system|> You are a clinical AI auditor model.\n"
        f"<|user|>\nInstruction: {instruction}\n\nReport:\n{report_text}\n"
        f"<|assistant|>\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=400,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = (
        response.replace("<|system|>", "")
                .replace("<|user|>", "")
                .replace("<|assistant|>", "")
                .strip()
    )
    return response


print("Running sample test...")
sample_report = """{
  "accuracy": 0.87,
  "precision": 0.82,
  "recall": 0.80,
  "f1_score": 0.81,
  "loss": 0.45
}"""

output = generate_response(sample_report)
print("\nAI Audit Output ===")
print(output)
