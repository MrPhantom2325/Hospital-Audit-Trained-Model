import os
import json
import re
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "PhantomAjusshi/phi3-auditor-merged"
JSONL_PATH = "data/processed/test.jsonl"

print("Loading model and tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
)

print("Model loaded successfully.")


def generate_response(report_text, instruction="Analyze the clinical model report and classify its health."):
    prompt = (
        f"<|system|> You are a clinical AI auditor model.\n"
        f"<|user|>\nInstruction: {instruction}\n\nReport:\n{report_text}\n"
        f"<|assistant|>\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = re.sub(r"<\|.*?\|>", "", response).strip()
    return response


def extract_category(text):
    match = re.search(r"Category:\s*([A-Za-z\s\(\)\-]+)", text)
    return match.group(1).strip() if match else "Unknown"


def evaluate_model(jsonl_path):
    if not os.path.exists(jsonl_path):
        print(f"File not found: {jsonl_path}")
        return

    true_labels, pred_labels, results = [], [], []
    print(f"Running evaluation on {jsonl_path}...\n")

    with open(jsonl_path, "r") as f:
        for idx, line in enumerate(f, start=1):
            data = json.loads(line)
            report_text = data["input"]
            true_label = extract_category(data["output"])

            response = generate_response(report_text)
            pred_label = extract_category(response)

            true_labels.append(true_label)
            pred_labels.append(pred_label)

            results.append({
                "input": report_text,
                "true_category": true_label,
                "predicted_category": pred_label,
                "response": response
            })

            print(f"{idx}. True: {true_label} | Predicted: {pred_label}")


    accuracy = accuracy_score(true_labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='weighted', zero_division=0)

    print("\n=== MODEL PERFORMANCE ===")
    print(f"Accuracy : {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall   : {recall:.3f}")
    print(f"F1-Score : {f1:.3f}")

    results_path = "inference_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "metrics": {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1},
            "predictions": results
        }, f, indent=2)

    print(f"\nResults saved to {results_path}")


evaluate_model(JSONL_PATH)
