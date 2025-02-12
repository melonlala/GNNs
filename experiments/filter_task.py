import os
import json

label_path = "results/math/labels.jsonl"

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

def main():
    labels = load_jsonl(label_path)
    tasks_dict = {}
    # tasks_dict = {"task1": {0: num_0, 1: num_1}, "task2": {0: num_0, 1: num_1}, ...}
    for record in labels:
        task = record["task"]
        label = record["label"]
        if task not in tasks_dict:
            tasks_dict[task] = {0: 0, 1: 0}
        tasks_dict[task][label] += 1
    
    # draw a histogram of the success rate of each task
    import matplotlib.pyplot as plt
    from collections import Counter
    success_rates = []
    for task, label_dict in tasks_dict.items():
        total = label_dict[0] + label_dict[1]
        try:
            # assert total == 280
            pass
        except:
            print(task, total)
        ratio = label_dict[1] / total
        success_rates.append(ratio)
    plt.hist(success_rates, bins=50)
    plt.xlabel("Success Rate")
    plt.ylabel("Number of Tasks")
    plt.title("Success Rate Distribution of Tasks")
    plt.savefig("results/math/workflow-label_pairs/success_rate_distribution.png")

    high_threshold = 0.8
    low_threshold = 0.2
    filtered_tasks = []
    for task, label_dict in tasks_dict.items():
        total = label_dict[0] + label_dict[1]
        ratio = label_dict[1] / total
        if ratio < high_threshold and ratio >= low_threshold:
            filtered_tasks.append(task)
    print(f"Number of tasks filtered >= {low_threshold} and < {high_threshold} : {len(filtered_tasks)}")

    filtered_labels = []
    lables_count = {"0": 0, "1": 0}
    for record in labels :
        if record["task"] in filtered_tasks:
            filtered_labels.append(record)
            lables_count[str(int(record["label"]))] += 1
            

    save_jsonl(filtered_labels, "results/math/workflow-label_pairs/filtered_labels.jsonl")
    print(f"{len(filtered_labels)}Filtered labels saved to results/math/workflow-label_pairs/filtered_labels.jsonl")
    print(f"Label 0: {lables_count['0']}, Label 1: {lables_count['1']}")

if __name__ == "__main__":
    main()