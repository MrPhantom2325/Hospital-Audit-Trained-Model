import json
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import get_peft_model, prepare_model_for_kbit_training
from lora_config import get_lora_config

model_name = "microsoft/Phi-3-mini-4k-instruct"
train_file = "data/processed/train.jsonl"
test_file = "data/processed/test.jsonl"
Output_dir = "models/phi3-auditor-lora"

max_len = 512
batch_size = 4
grad_accum = 4
epochs = 3
lr = 1e-4


def load_processed_dataset():
    return load_dataset(
        "json",
        data_files={"train": train_file, "test": test_file}
    )

def apply_format(example):

    system_prompt = "You are an AI auditor analyzing clinical model performance reports."
    
    text = (
        f"<|system|>\n{system_prompt}\n"
        f"<|user|>\nInstruction: {example['instruction']}\n\nReport:\n{example['input']}\n"
        f"<|assistant|>\n{example['output']}"
    )

    return text


def tokenize(batch, tokenizer):
    merged_text = [apply_format(x) for x in batch]
    return tokenizer(
        merged_text,
        truncation=True,
        padding="max_length",
        max_length=max_len
    )


def main():
    print("\n=== Loading tokenizer & model ===")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        load_in_4bit=True
    )

    print("Preparing model for QLoRA…")
    model = prepare_model_for_kbit_training(model)

    print("Loading LoRA config…")
    lora_cfg = get_lora_config()
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    print("Loading dataset…")
    dataset = load_processed_dataset()

    print("Tokenizing dataset…")
    tokenized = dataset.map(
        lambda batch: tokenize(batch, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    training_args = TrainingArguments(
        output_dir=Output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        num_train_epochs=epochs,
        warmup_ratio=0.1,
        logging_steps=25,
        evaluation_strategy="steps",
        eval_steps=200,
        save_steps=200,
        fp16=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=data_collator,
    )

    print("\n=== Training Started ===")
    trainer.train()

    print("\n=== Saving Adapter ===")
    model.save_pretrained(Output_dir)
    tokenizer.save_pretrained(Output_dir)

    print("\nTraining complete!")

if __name__ == "__main__":
    main()
