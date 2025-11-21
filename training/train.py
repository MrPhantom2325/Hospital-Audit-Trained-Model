import json
from pathlib import Path
import math
import torch
from datasets import load_dataset
from transformers import (
  Autotokenizer,
  AutomodelForCausalLM,
  TrainingArguments,
  Trainer,
  DataCollatorForLanguageModeling
 )
from peft import get_peft_model, prepare_model_for_kbit_training, set_peft_model_state_dict
from lora_config import get_lora_config

model_name = "microsoft/Phi-3-mini-4k-instruct"
train_file = "data/processed/train.jsonl"
test_file = "data/processed/test.jsonl"
output_dir= "models/lora/phi3-mini-lora"

Max_length = 512
batch_size = 4
grad_accum = 4
epochs = 3
learning_rate = 1e-4

def load_processed_dataset():
    dataset=load_dataset("json", data_files=
                         { "train": train_file,
                           "test": test_file} )
    return dataset

def tokenize_function(examples,tokenizer):
    instructions_data = examples["instruction"]
    input_data = examples["input"]
    output_data = examples["output"]

    text=[
        f"<|system|> You are an AI model analyzing clinical audit reports.\n"
        f"<|user|>\n{instruction}\n\nReport:\n{input_data}\n"
        f"<|assistant|>\n{output_data}"
    ]

    return tokenizer(
        text,
        truncation=True,
        max_length=Max_length,
        padding="max_length"
    )

def main():
    print("Loading tokenizer and model:", model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,
        device_map="auto"
    )

    print("Preparing model for QLoRA…")
    model = prepare_model_for_kbit_training(model)

    lora_config = get_lora_config()
    print("Using LoRA config:", lora_config)

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Loading dataset…")
    dataset = load_processed_dataset()

    print("✏ Tokenizing dataset…")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=False
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        warmup_ratio=0.1,
        logging_steps=20,
        save_steps=200,
        evaluation_strategy="steps",
        eval_steps=200,
        fp16=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator
    )

    print("Starting training…")
    trainer.train()

    print("Saving LoRA adapter to:", output_dir)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("Training complete!")


if __name__ == "__main__":
    main()
