import os
import json
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="AFlow GSM8K task_filter")
    parser.add_argument("--dataset_path", type=str, default="metagpt/ext/aflow/data/gsm8k_validate.jsonl")
    parser.add_argument("--results_dir", type=str, default="experiments/GSM8K/labels_latest")
    parser.add_argument("--root_dir", type=str, default="experiments/GSM8K")
    parser.add_argument("--filtered_questions_path", type=str, default="experiments/GSM8K/filtered_questions.jsonl")
    return parser.parse_args() 
    
def load_dataset(path):
    dataset = []
    with open(path, "r") as f:
        for line in f:
            data = json.loads(line)
            dataset.append(data)
    return dataset

def get_task_id(question, dataset_path):
    dataset = load_dataset(dataset_path)
    for data in dataset:
        if data["question"] == question:
            return data["question"]
    return None

def save_records(records, path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
        
    with open(path, "w") as f:
        json.dump(records, f, indent=4)
    
def get_records(dataset_path, results_dir, record_save_path):
    num_workflows = 0
    dataset = load_dataset(dataset_path)
    num_tasks = len(dataset)
    task_records = {}
    sub_dirs = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
    for sub_dir in sub_dirs:
        if not os.path.isdir(os.path.join(results_dir, sub_dir)):
            continue
        for file_name in os.listdir(os.path.join(results_dir, sub_dir)):
            if file_name.endswith(".json") and file_name.startswith("0."):
                with open(os.path.join(results_dir, sub_dir, file_name), "r") as f:
                    data = json.load(f)
                num_workflows += 1
                break   
            
        try:
            assert num_tasks == len(data)
        except:
            print(f"Error: {sub_dir} has {len(data)} tasks, but should have {num_tasks}.")
            continue
            
        # filter out tasks with diverse scores on different workflows
        for sample in data:
            question = sample["question"]
            score = sample["score"]
            task_id = get_task_id(question, dataset_path)
            if task_id is None:
                print(f"Error: {question} not found in dataset.")
                continue    
            workflow_id = sub_dir.split("_")[-1]
            if task_id not in task_records:
                task_records[task_id] = [{"scores": score, "workflows": workflow_id}]
                # task_records[task_id] = [score]
                # print(f"Task {task_id} added with score {score} on workflow {workflow_id}.")
            else:
                task_records[task_id].append({"scores": score, "workflows": workflow_id})
                # task_records[task_id].append(score)
                # print(f"Task {task_id} updated with score {score} on workflow {workflow_id}.")
    print("recorded tasks number", len(task_records))
    save_records(task_records, record_save_path)
    return record_save_path


def read_records(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def get_workflow_len(results_dir):
    num_workflows = 0
    for sub_dir in os.listdir(results_dir):
        if not os.path.isdir(os.path.join(results_dir, sub_dir)):
            continue
        num_workflows += 1
    
    return num_workflows

def get_score_features(record_path):
    records = read_records(record_path)
    num_tasks = 0
    avg_scores_dict = {}
    score_variance_dict = {}
    for task_id, samples in records.items():
        # if len(samples) != (num_workflows-2):
        #     print(f"Task {task_id} has {len(samples)} samples, but should have {num_workflows-2}.")
        #     continue
        avg_score = sum([sample["scores"] for sample in samples])/len(samples)
        # count the 0/1 label rate
        label_count = {0: 0, 1: 0}
        for sample in samples:
            label_count[sample["scores"]] += 1
        score_variance = label_count[0]/len(samples)
        avg_scores_dict[task_id] = avg_score
        score_variance_dict[task_id] = score_variance
        num_tasks += 1

    print(f"Number of tasks: {num_tasks}")
    # with open(avg_scores_path, "w") as f:
    #     json.dump(avg_scores_dict, f, indent=4)
    # with open(score_variance_path, "w") as f:
    #     json.dump(score_variance_dict, f, indent=4)
    return avg_scores_dict, score_variance_dict
        
# get_score_features(modified_path, score_variance_path, avg_scores_path)

def filter_records(threshold, score_variance_dict, fig_path):
    print(f"Number of tasks after filtering: {len(score_variance_dict)}")
    # 计算最大值和最小值
    max_variance = max(score_variance_dict.values(), default=0.0)
    min_variance = min(score_variance_dict.values(), default=0.0)
    print(f"Max variance: {max_variance}, Min variance: {min_variance}")
    
    variances = list(score_variance_dict.values())
    plt.hist(variances, bins=50, edgecolor='black')
    plt.xlabel("Score Variance")
    plt.ylabel("Number of Tasks")
    plt.title("Distribution of Score Variance")
    plt.savefig(fig_path)
    # count the last 2 bins tasks
    
    print(f"Threshold: {threshold}")
    count = 0
    for variance in variances:
        if variance > threshold:
            count += 1
            
    print(f"Number of tasks with variance greater than {threshold}: {count}")

# filter_records(modified_path, score_variance_path, fig_path)

def save_filter_labels(label_path, score_var_dict, threhold, filter_labels_path):
    
    records = []
    with open(label_path, "r") as f:
        for line in f:
            records.append(json.loads(line))
    filtered_labels = []
    filter
    for sample in records:
        question = sample["task"]
        if score_var_dict[question] > threhold:
            filtered_labels.append(sample)
    
    # save as jsonl
    with open(filter_labels_path, "w") as f:
        for label in filtered_labels:
            f.write(json.dumps(label) + "\n")
    print(f"Number of filtered labels: {len(filtered_labels)}")
    #count label 0 and 1
    label_count = {0: 0, 1: 0}
    for label in filtered_labels:
        label_count[label["label"]] += 1
    print(f"Label 0: {label_count[0]}, Label 1: {label_count[1]}")

def filter_records_from_questions_file(questions_path, label_path, filtered_label_path):
    with open(questions_path, "r") as f:
        questions = json.load(f)
        
    with open(label_path, "r") as f:
        labels = []
        for line in f:
            labels.append(json.loads(line))
    
    count_label = {0: 0, 1: 0}
    filtered_labels = []
    for label in labels:
        if label["task"] in questions:
            filtered_labels.append(label)
            count_label[label["label"]] += 1
    # save as jsonl
    with open(filtered_label_path, "w") as f:
        for label in filtered_labels:
            f.write(json.dumps(label) + "\n")
    print(f"Number of filtered labels: {len(filtered_labels)}")
    print(f"Label 0: {count_label[0]}, Label 1: {count_label[1]}")

if __name__ == "__main__":
    args = parse_args()
    fig_path = f"{args.root_dir}/score_variance_distribution.png"
    record_save_path = f"{args.root_dir}/records.json"
    label_path = f"{args.root_dir}/labels.jsonl"
    # record_save_path = get_records(args.dataset_path, args.results_dir, record_save_path)
    print("record saved in", record_save_path)
    # avg_scores_dict, score_variance_dict = get_score_features(record_save_path)
    # # print(score_variance_dict[0])
    # threshold = 0.3
    # filter_records(threshold, score_variance_dict, fig_path)

    filter_labels_path = f"{args.root_dir}/filtered_labels.jsonl"
    # save_filter_labels(label_path, score_variance_dict, threshold, filter_labels_path)

    filter_records_from_questions_file(args.filtered_questions_path, label_path, filter_labels_path)

            
            
    

    



        
                
    
        
    

