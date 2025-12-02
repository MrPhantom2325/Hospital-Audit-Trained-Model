import json
from datasets import load_dataset
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import get_peft_model, prepare_model_for_kbit_training
from lora_config import get_lora_config


MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

TRAIN_FILE = "data/processed/train.jsonl"
TEST_FILE = "data/processed/test.jsonl"
OUTPUT_DIR = "models/phi3-auditor-lora-8bit"

MAX_LEN = 512
BATCH_SIZE = 4
GRAD_ACCUM = 4
EPOCHS = 3
LR = 1e-4


def format_example(instruction, report_json, output_text):

    system_prompt = "You are an AI auditor analyzing clinical model performance reports."

    return (
        f"<|system|>\n{system_prompt}\n"
        f"<|user|>\nInstruction: {instruction}\n\nReport:\n{report_json}\n"
        f"<|assistant|>\n{output_text}"
    )


def tokenize_batch(batch, tokenizer):
    texts = [
        format_example(instr, inp, outp)
        for instr, inp, outp in zip(batch["instruction"], batch["input"], batch["output"])
    ]

    return tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
    )


def main():
    print("\n=== Loading tokenizer ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("=== Loading base model in 8-bit with LoRA ===")
    quant_config = BitsAndBytesConfig(
        load_in_8bit=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
        device_map="auto",
    )


    model = prepare_model_for_kbit_training(model)

    print("=== Applying LoRA config ===")
    lora_cfg = get_lora_config()
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    print("=== Loading JSONL dataset ===")
    dataset = load_dataset(
        "json",
        data_files={"train": TRAIN_FILE, "test": TEST_FILE}
    )

    print("=== Tokenizing dataset ===")
    tokenized = dataset.map(
        lambda batch: tokenize_batch(batch, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    print("=== Setting up training args ===")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        warmup_ratio=0.1,
        logging_steps=50,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=data_collator,
    )

    print("\n=== Training started ===")
    trainer.train()

    print("\n=== Saving LoRA adapter + tokenizer ===")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\n Training complete! LoRA 8-bit model saved at:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
