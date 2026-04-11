# 🏥 Hospital Audit Trained Model — Phi-3 LoRA Fine-tune

A fine-tuned clinical AI auditing model built on **Microsoft Phi-3-mini-4k-instruct** using **LoRA (Low-Rank Adaptation)** with **8-bit quantization**. Given a JSON report of clinical ML model performance metrics, the model classifies the model's health and generates a human-readable explanation.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Dataset](#dataset)
- [Model Details](#model-details)
- [Training](#training)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Merging LoRA Weights](#merging-lora-weights)
- [Requirements](#requirements)

---

## Overview

Clinical ML models deployed in hospital settings need continuous monitoring. This project automates that auditing process by fine-tuning Phi-3-mini to act as a **clinical AI auditor** — it reads a model's performance metrics (AUC, ECE, drift, label shift, etc.) and returns a structured audit verdict with a category and explanation.

**Example output:**
```
Category: Calibration Failure
Explanation: High calibration error (ECE 0.2781) despite reasonable discrimination (AUC 0.863).
             Recommend recalibration (Platt scaling / isotonic) and threshold review.
```

---

## Project Structure

```
Hospital-Audit-Trained-Model/
│
├── data/
│   ├── raw/
│   │   └── audit_dataset_v2_5000.json       # Raw dataset (5000 samples)
│   ├── processed/
│   │   ├── train.jsonl                       # 80% training split
│   │   └── test.jsonl                        # 20% test split
│   └── templates/
│       ├── prompt_template.txt               # Prompt format template
│       └── response_template.txt             # Response format template
│
├── training/
│   ├── dataset_builder.py                    # Builds train/test JSONL from raw data
│   ├── lora_config.py                        # LoRA hyperparameter configuration
│   └── train.py                              # Main training script
│
├── inference/
│   ├── run_inference.py                      # Single sample inference
│   └── Metrics_Test.py                       # Batch evaluation on test set
│
├── models/
│   └── phi3-auditor-lora-8bit/              # Saved LoRA adapter + tokenizer
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       ├── tokenizer files...
│       ├── checkpoint-500/
│       └── checkpoint-675/
│
└── merge_lora.py                             # Merges LoRA adapter into base model
```

---

## How It Works

1. **Input:** A JSON object containing clinical ML model metrics (AUC, accuracy, precision, recall, F1, ECE, Brier score, drift, missing rate, label shift, etc.)
2. **Prompt:** The metrics are injected into a structured prompt with a system instruction for the auditor role.
3. **Output:** The model returns a `Category:` classification and an `Explanation:` with actionable recommendations.

**Prompt format:**
```
<|system|>
You are an AI auditor analyzing clinical model performance reports.
<|user|>
Instruction: Analyze the clinical model report and classify its health.

Report:
{ ...metrics JSON... }
<|assistant|>
```

---

## Dataset

- **Source:** `audit_dataset_v2_5000.json` — a synthetic dataset of 5,000 clinical model audit reports generated on `2025-11-17`.
- **Fields per record:** `metrics` (JSON object), `audit_label` (category string), `explanation` (natural language justification).
- **Split:** 80/20 train/test using `sklearn.model_selection.train_test_split` with `random_state=42`.
- **Processed format:** JSONL with three fields — `instruction`, `input` (metrics JSON), `output` (category + explanation).

**To rebuild the processed dataset:**
```bash
cd training
python dataset_builder.py
```

---

## Model Details

| Property | Value |
|---|---|
| Base Model | `microsoft/Phi-3-mini-4k-instruct` |
| Fine-tuning Method | LoRA (PEFT) |
| Training Quantization | 8-bit (`BitsAndBytesConfig`) |
| Merged Model Precision | FP16 (F16 safetensors) |
| Model Size | ~4B parameters |
| Model Weight Files | 2 shards (`model-00001-of-00002.safetensors` + `model-00002-of-00002.safetensors`) |
| Total Repo Size | 7.65 GB |
| PEFT Version | 0.18.0 |
| LoRA Rank (`r`) | 16 |
| LoRA Alpha | 32 |
| LoRA Dropout | 0.05 |
| Target Modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Task Type | `CAUSAL_LM` |
| HuggingFace Model | [PhantomAjusshi/phi3-auditor-merged](https://huggingface.co/PhantomAjusshi/phi3-auditor-merged) |

---

## Training

**Hyperparameters:**

| Parameter | Value |
|---|---|
| Epochs | 3 |
| Batch Size | 4 |
| Gradient Accumulation Steps | 4 |
| Effective Batch Size | 16 |
| Learning Rate | 1e-4 |
| Warmup Ratio | 0.1 |
| Max Sequence Length | 512 |
| Precision | FP16 (if CUDA available) |
| Checkpointing | Every 500 steps |

**Training loss progression:**

| Step | Epoch | Loss |
|---|---|---|
| 50 | 0.22 | 1.6228 |
| 100 | 0.44 | 0.6565 |
| 150 | 0.67 | 0.4436 |
| 500 | 2.22 | 0.4109 |
| 675 | 3.00 | ~0.410 |

**To train from scratch:**
```bash
cd training
python train.py
```

> **Note:** Requires a CUDA-enabled GPU with at least 8GB VRAM for 8-bit training.

---

## Inference

Run a single inference using the merged model from HuggingFace:

```bash
python inference/run_inference.py
```

The script loads the model directly from HuggingFace at `PhantomAjusshi/phi3-auditor-merged`. You can also use it programmatically:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "PhantomAjusshi/phi3-auditor-merged"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
    trust_remote_code=True   # required for Phi-3 custom modeling code
)

report = """{
  "auc": 0.863,
  "accuracy": 0.83,
  "precision": 0.79,
  "recall": 0.69,
  "f1": 0.79,
  "ece": 0.278,
  "brier": 0.263,
  "drift": 0.03,
  "missing_rate": 0.003,
  "label_shift": 0.06,
  "pos_rate": 0.10,
  "data_integrity_issues": 0
}"""

prompt = (
    f"<|system|> You are a clinical AI auditor model.\n"
    f"<|user|>\nInstruction: Analyze the clinical model report and classify its health.\n\nReport:\n{report}\n"
    f"<|assistant|>\n"
)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.inference_mode():
    outputs = model.generate(**inputs, max_new_tokens=400, temperature=0.7, top_p=0.9, repetition_penalty=1.2)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

> **Note:** `trust_remote_code=True` is required because the HuggingFace repo includes custom Phi-3 modeling files (`modeling_phi3.py`, `configuration_phi3.py`).

**Generation settings:** `max_new_tokens=400`, `temperature=0.7`, `top_p=0.9`, `repetition_penalty=1.2`

---

## Evaluation

Run batch evaluation on the test set to get classification metrics:

```bash
python inference/Metrics_Test.py
```

This generates predictions for all records in `data/processed/test.jsonl`, computes weighted precision, recall, F1, and accuracy against ground-truth `Category:` labels, and saves full results to `inference_results.json`.

---

## Merging LoRA Weights

To merge the LoRA adapter into the base model for standalone deployment (no PEFT dependency at inference time):

```bash
python merge_lora.py
```

Output is saved to `models/phi3-auditor-merged/`. The merged model is published on HuggingFace as [PhantomAjusshi/phi3-auditor-merged](https://huggingface.co/PhantomAjusshi/phi3-auditor-merged) and consists of:

- `model-00001-of-00002.safetensors` (4.97 GB)
- `model-00002-of-00002.safetensors` (2.67 GB)
- `modeling_phi3.py` + `configuration_phi3.py` (custom Phi-3 code, required for `trust_remote_code=True`)
- Tokenizer files (`tokenizer.json`, `tokenizer.model`, `tokenizer_config.json`, etc.)

> **Note:** Large model files (`.safetensors`, `.bin`, `.pt`) are excluded from this repository via `.gitignore`. Use the HuggingFace hosted weights directly or retrain and merge locally.

---

## Requirements

```bash
pip install torch transformers peft bitsandbytes datasets scikit-learn accelerate
```

| Package | Role |
|---|---|
| `transformers` | Base model loading, tokenizer, training |
| `peft` | LoRA adapter configuration and application |
| `bitsandbytes` | 8-bit quantization |
| `datasets` | JSONL dataset loading |
| `scikit-learn` | Train/test split and evaluation metrics |
| `accelerate` | Device mapping and mixed precision |

---

## License

This project is for research and educational purposes. The base model is subject to Microsoft's [Phi-3 license](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct).
