import json
import os


with open('datasets/humaneval/chosen_tasks.jsonl', 'r') as f:
    for line in f:
        data = [json.loads(line) for line in f]

unique_tasks = {}
for item in data:
    name = item['name']
    if name not in unique_tasks:
        unique_tasks[name] = 1
    else:
        unique_tasks[name] += 1

for task, count in unique_tasks.items():
    print(f'{task}: {count}')