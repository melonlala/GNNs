import os
import json

results_dir = "results/math/workflow-label_pairs"
total_label = 0
count_finished = 0
total_num = 0
# delete repeating questions in result_file
task_dict = {}

for subdir in os.listdir(results_dir):
    if not os.path.isdir(f"{results_dir}/{subdir}"):
        continue

    for file in os.listdir(f"{results_dir}/{subdir}"):
        if file.endswith(".json"):
            with open(f"{results_dir}/{subdir}/{file}", "r") as f:
                resume_data = json.load(f)
        total_num += 1
        current_questions = []
        count_removed = 0
        new_data = []
        for data in resume_data:
            if data["task"]["problem"] not in current_questions:
                current_questions.append(data["task"]["problem"])
                new_data.append(data)
                total_label += 1
                if data["task"]["problem"] not in task_dict:
                    task_dict[data["task"]["problem"]] = 1
                else:
                    task_dict[data["task"]["problem"]] += 1
            else:
                count_removed += 1
            
            
        # print(f"lenge of {file}: {len(new_data)}")
        if len(new_data) == 605:
            count_finished += 1
            with open(f"{results_dir}/{subdir}/{file}", "w") as f:
                json.dump(new_data, f, indent=4)
        else:
            # os.remove(f"{results_dir}/{subdir}/{file}")
            print(f"Removed {subdir}{file}")
        if count_removed > 0:
            print(f"Removed {count_removed} repeating questions in {subdir}{file}")

# find 605 number's task in task_dict
shared_task = []
for key, value in task_dict.items():
    if value == total_num:
        shared_task.append(key)
print(f"Shared task: {len(shared_task)}")
print(f"Total finished: {count_finished}")
print(f"Total num: {total_num}")
print(f"Total label: {total_label}")