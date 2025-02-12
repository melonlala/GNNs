import json
import os

def jsonl2json(jsonl_file, json_file):
    with open(jsonl_file, 'r') as f:
        lines = f.readlines()
        data = []
    for line in lines:
        data.append(json.loads(line))
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == '__main__':
    data_dir = './datasets/humaneval/filter_results_2024-11-14-14-59-42'
    jsonl_file = f"{data_dir}/all_false_tasks.jsonl"
    json_file = f"{data_dir}/all_false_tasks.json"
    jsonl2json(jsonl_file, json_file)