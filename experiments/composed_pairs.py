import os
import json
def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def save_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for record in data:
            json.dump(record, f, ensure_ascii=False)
            f.write('\n')

def check_tasks_number(pairs):
    task_dict = {}
    for record in pairs:
        if record["task"]["problem"] not in task_dict:
            task_dict[record["task"]["problem"]] = 1
        else:
            return False
    if len(task_dict) != 427:
        print("length of task_dict: ", len(task_dict))
    return len(task_dict) == 427
# 定义基础目录和标签目录
base_dir ="results/mmath/workflow-label_pairs"
labels_dir = os.path.join(base_dir, "workflow-label_pairs")
save_path = os.path.join(base_dir, "labels.jsonl")

# 确保标签目录存在
if not os.path.exists(labels_dir):
    os.makedirs(labels_dir)

# 初始化保存结果的列表
labels = []
labels_path = []
# 遍历 base_dir 下的所有子目录
for sub_dir in os.listdir(base_dir):
    sub_dir_path = os.path.join(base_dir, sub_dir)
    if os.path.isdir(sub_dir_path):  # 确保是目录
        # 遍历子目录中的所有文件
        for file_name in os.listdir(sub_dir_path):
            file_path = os.path.join(sub_dir_path, file_name)
            if os.path.isfile(file_path) and file_name.endswith(".json"):  # 确保是 JSON 文件

                labels_path.append(file_path)
print("length of labels_path: ", len(labels_path))
count = 0
avg_score_list = []
for file_path in labels_path:
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            pairs = json.load(f)
        except:
            print("error: ", file_path)
            
        # 假设每个 JSON 文件中包含一个字典，包含 'workflow' 和 'label' 键
        check = check_tasks_number(pairs)
        if not check:
            # print("error: ", file_path)
            continue
        count += 1
        avg_score = 0
        for record in pairs:
            # print(record)
            update_item = {
                "task": record["task"]["prompt"],
                "nodes": record["nodes"],
                "edge_index": record["edge_index"],
                "label": record["score"]
            }
            labels.append(update_item)
            avg_score += record["score"]
        avg_score = avg_score / len(pairs)
        avg_score_list.append(avg_score)
print("avg_score_list: ", avg_score_list)
print("count: ", count)
print("length of labels: ", len(labels))
# 将结果保存到 labels.jsonl 文件中
with open(save_path, 'w', encoding='utf-8') as f:
    for label in labels:
        json.dump(label, f, ensure_ascii=False)
        f.write('\n')  # 每个 JSON 对象占一行

print(f"Labels saved to {save_path}")
