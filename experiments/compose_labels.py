import os
import json


base_dir = "/home/ubuntu/DATA2/yuchenhou/GNN/GDesigner-3063/results/math"
labels_dir = os.path.join(base_dir, "workflow-label_pairs")
save_path = os.path.join(base_dir, "labels.jsonl")





labels = []
count_finished = 0
count_label = {0: 0, 1: 0}
# 遍历 base_dir 下的所有子目录
for sub_dir in os.listdir(labels_dir):
    sub_dir_path = os.path.join(labels_dir, sub_dir)
    if os.path.isdir(sub_dir_path):  # 确保是目录
        # 遍历子目录中的所有文件
        for file_name in os.listdir(sub_dir_path):
            file_path = os.path.join(sub_dir_path, file_name)
            if os.path.isfile(file_path) and file_name.endswith(".json"):  # 确保是 JSON 文件
                with open(file_path, 'r', encoding='utf-8') as f:
                    pairs = json.load(f)
                    # if len(pairs) != 427:
                    #     continue
                    # 假设每个 JSON 文件中包含一个字典，包含 'workflow' 和 'label' 键
                    count_finished += 1
                    for record in pairs:
                        # print(record)
                        update_item = {
                            "task": record["task"]["problem"],
                            "nodes": record["nodes"],
                            "edge_index": record["edge_index"],
                            "label": record["score"]
                        }
                        count_label[int(record["score"])] += 1
                        labels.append(update_item)
# 将结果保存到 labels.jsonl 文件中
with open(save_path, 'w', encoding='utf-8') as f:
    for label in labels:
        json.dump(label, f, ensure_ascii=False)
        f.write('\n')  # 每个 JSON 对象占一行

print(f"Labels saved to {save_path}")
print(f"Total finished: {count_finished}")
print(f"Total label: {count_label}")