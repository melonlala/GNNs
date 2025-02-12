import json
import os

dataset_file = "metagpt/ext/aflow/data/mbpp_test.jsonl"
from_file = "experiments/MBPP/workflows_latest/round_38/0.94554_20250118_165027.json"

def load_jsonl(file):
    data = []
    with open(file, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def load_json(file):
    with open(file, 'r', encoding='utf-8') as file:
                data = json.load(file)
    return data

def save_jsonl(data, file):
    with open(file, 'w') as f:
        for line in data:
            f.write(json.dumps(line) + '\n')

def filter_data(dataset, records):
    filtered_data = []
    for data in dataset:
        for record in records:
            if data["prompt"] == record["inputs"]:
                filtered_data.append(data)
                break
    return filtered_data

if __name__ == "__main__":
    data = load_jsonl(dataset_file)
    records = load_json(from_file)
    filter_data = filter_data(data, records)
    save_jsonl(filter_data, "metagpt/ext/aflow/data/mbpp_filtered.jsonl")
    print(f"Number of filtered records: {len(filter_data)}")
    print(f"Filtered records saved to metagpt/ext/aflow/data/mbpp_filtered.jsonl")

