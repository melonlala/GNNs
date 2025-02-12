import os
import json
import shutil

ori_dir = "experiments/MATH/GSM8K_workflow_results"
new_dir = "experiments/MATH/GSM8K_workflows"

count = 0
# replace the score_json file in new_dir with the one in ori_dir
for sub_dir in os.listdir(new_dir):
    if not os.path.isdir(os.path.join(new_dir, sub_dir)):
        continue
    count += 1
    for file in os.listdir(os.path.join(new_dir, sub_dir)):
        if (file.startswith("0") or file.startswith("1")) and file.endswith(".json"):
            # remove file
            # os.remove(os.path.join(new_dir, sub_dir, file))
            print(f"Removed {file}")
print(f"Removed {count} dirs.")

count = 0
for sub_dir in os.listdir(new_dir):
    if not os.path.isdir(os.path.join(new_dir, sub_dir)):
        continue
    count += 1
    ori_sub_dir = os.path.join(ori_dir, sub_dir)
    new_sub_dir = os.path.join(new_dir, sub_dir)
    for file in os.listdir(ori_sub_dir):
        if (file.startswith("0") or file.startswith("1")) and file.endswith(".json"):
            ori_file = os.path.join(ori_sub_dir, file)
            new_file = os.path.join(new_sub_dir, file)
            # replace the same name file in new_dir with the one in ori_dir
            shutil.copy(ori_file, new_file)
            print(f"Replaced {new_file} with {ori_file}")
print(f"Replaced {count} dirs.")
            
    