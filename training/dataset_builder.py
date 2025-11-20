import json
from pathlib import Path
from sklearn.model_selection import train_test_split

def load_templates():
    prompt_template = Path("data/templates/prompt_template.txt").read_text()
    response_template = Path("data/templates/response_template.txt").read_text()
    return prompt_template, response_template

def load_raw():
    raw_path = Path("data/raw/audit_dataset_v2_5000.json")
    raw = json.loads(raw_path.read_text())
    return raw["source_1_versioned_dataset"]["train"]   

def process_data():
    prompt_template, response_template = load_templates()
    dataset = load_raw()
    processed = []

    for i in dataset:   
        report_json = json.dumps(i["metrics"], indent=2)

        response = response_template.replace("{audit_label}", i["audit_label"]).replace("{explanation}", i["explanation"])

        processed.append({
            "instruction": "Analyze the clinical model report and classify its health.",
            "input": report_json,
            "output": response
        })

    return processed

def save_jsonl(data, path):
    with open(path, "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")

def main():
    processed = process_data()
    train, test = train_test_split(processed, test_size=0.2, random_state=42)

    save_jsonl(train, "data/processed/train.jsonl")
    save_jsonl(test, "data/processed/test.jsonl")

    print("Created:", len(train), "training samples")
    print("Created:", len(test), "test samples")

if __name__ == "__main__":
    main()
