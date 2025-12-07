import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
LORA_ADAPTER_PATH = "models/phi3-auditor-lora"
MERGED_OUTPUT_PATH = "models/phi3-auditor-merged"

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, device_map="auto")

print("Attaching LoRA adapter...")
model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)

print("Merging LoRA weights into the base model...")
model = model.merge_and_unload()

print("Saving merged model to:", MERGED_OUTPUT_PATH)
model.save_pretrained(MERGED_OUTPUT_PATH)

print("Saving tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.save_pretrained(MERGED_OUTPUT_PATH)

print("Merge complete! Load the model from:", MERGED_OUTPUT_PATH)
