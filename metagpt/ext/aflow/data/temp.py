import os
import json

source_file = "./MetaGPT/metagpt/ext/aflow/data/humaneval_chosen_tasks.jsonl"
data_file = "./MetaGPT/metagpt/ext/aflow/data/humaneval_full.jsonl"
save_file = "./MetaGPT/metagpt/ext/aflow/data/humaneval_validate.jsonl"

task_ids = set()
with open(source_file, "r") as f:
    source_data = [json.loads(line) for line in f]
    for item in source_data:
        task_id = item["name"].split("_")[1]
        task_ids.add(task_id)
    
with open(data_file, "r") as f:
    data = [json.loads(line) for line in f]

save_data = []
for item in data:
    task_id = item["task_id"].split("/")[-1]
    if task_id in task_ids:
        save_data.append(item)
        
with open(save_file, "w") as f:
    for item in save_data:
        f.write(json.dumps(item) + "\n")
    