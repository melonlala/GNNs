import os
import json

base_dir = "./metagpt/ext/aflow/scripts/optimized/HumanEval/workflows"
# analysis_dir = "./experiments/HuamnEval/mod_updated_workflows"
save_dir = "./experiments/HuamanEval/workflows"
os.makedirs(save_dir, exist_ok=True)
subdirs = [os.path.join(base_dir, o) for o in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir,o)) and o.startswith("round")]

from collections import defaultdict

def process_json_files(subdirs, record_dict, file_count):
    for subdir in subdirs:
        count_0 = 0
        count_1 = 0
        cost = 0
        json_files = [
            os.path.join(subdir, o)
            for o in os.listdir(subdir)
            if os.path.isfile(os.path.join(subdir, o)) and
               o.endswith(".json") and
               o not in ["log.json", "experience.json"]
        ]
        
        if not json_files:
            print(f"No valid JSON files found in {subdir}. Skipping...")
            continue
        
        file = json_files[0]
        avg_score = float(file.split("/")[-1].split("_")[0])
        if avg_score == 0.0:
            continue
        # os.system(f"cp -r {subdir} {save_dir}/")
        with open(file, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                file_count += 1
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in file {file}: {e}")
                continue
        
        for item in data:
            try:
                task_name = item["inputs"]
                score = float(item["score"])
                cost += item["cost"]
            except KeyError as e:
                print(f"Missing key in item: {e}. Skipping item...")
                continue
            except ValueError:
                print(f"Invalid score value: {item.get('score')}. Skipping item...")
                continue
            
            record_dict[task_name][0] += score
            record_dict[task_name][1] += 1
            
            if score == 0.0:
                count_0 += 1
            else:
                count_1 += 1

    return record_dict, file_count, count_0, count_1, cost

def calculate_success_rates(record_dict):
    success_rates = {}
    for task_name, record in record_dict.items():
        if record[1] == 0:
            success_rate = 0
        else:
            success_rate = record[0] / record[1]
        success_rates[task_name] = success_rate
    return success_rates

file_count = 0
total_count_0 = 0
total_count_1 = 0
total_cost = 0
record_dict = defaultdict(lambda: [0.0, 0])
for subdir in subdirs:
    print(f"Processing {subdir}...")
    # if subdir in analysis_dir:
    if os.path.exists(os.path.join(save_dir, subdir.split("/")[-1])):
        print(f"{subdir} has already been processed. Skipping...")
        continue
    record_dict, file_count, count_0, count_1, cost = process_json_files([subdir], record_dict, file_count)
    total_count_0 += count_0
    total_count_1 += count_1
    total_cost += cost

print(f"Total count of 0: {total_count_0}")
print(f"Total count of 1: {total_count_1}")
print(f"Total cost: {total_cost}")
print(f"Processed {file_count} files.")
success_rates = calculate_success_rates(record_dict)
# draw bar chart, x is the thread of success rate, y is the number of tasks
# sorted by success rate in descending order
import matplotlib.pyplot as plt
success_rates_values = list(success_rates.values())
num_bins = 10
counts, bins, patches = plt.hist(success_rates_values, bins=num_bins, color='skyblue', edgecolor='black')

plt.xlabel("Success Rate")
plt.ylabel("Number of Tasks")
plt.title("Distribution of Success Rates in AFLOW Optimized MMLU Workflows")
plt.tight_layout()

# 添加数据标签
for count, x_val in zip(counts, bins):
    if count > 0:
        plt.text(x_val, count, f"{int(count)}", ha='center', va='bottom', fontsize=8)

plt.savefig("success_rate.png")
        
    