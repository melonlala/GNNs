import os
import json
import numpy as np
import glob
import pandas as pd

path = "./metagpt/ext/aflow/data/mmlu_validate.jsonl"
task_path = "./metagpt/ext/aflow/data/mmlu_tasks.jsnol"
dir_path = "./datasets/mmlu/test/"
full_questions_path = "./datasets/mmlu/full_questions.jsonl"
save_path = "./metagpt/ext/aflow/data/mmlu.jsonl"

def get_tasks(path, save_path):
    with open(path, "r") as f:
        data = [json.loads(line) for line in f]

    # save unique tasks in the data
    tasks = []
    count = 0
    for i, d in enumerate(data):
        new_task = d["task"]
        if new_task not in tasks:
            tasks.append(new_task)
            count += 1

    with open(save_path, "w") as f:
        for task in tasks:  
            f.write(json.dumps({"task": task}) + "\n")
            
def collect_full_questions(dir_path, save_path):
    rng = np.random.default_rng(888)
    csv_paths = glob.glob(dir_path + "*.csv")
    csv_paths = sorted(csv_paths)
    print("Number of topics: ", len(csv_paths))

    names = ['question', 'A', 'B', 'C', 'D', 'correct_answer']

    total_df = pd.DataFrame(columns=names)
    for path in csv_paths:
        single_df = pd.read_csv(path, header=None,
                        names=names,encoding='utf-8')
        total_df = pd.concat([total_df, single_df])

    total_df = total_df.reset_index(drop=True)

    # Pseudorandom shuffle
    total_df = total_df.reindex(rng.permutation(total_df.index))

    print("Total number of questions: ", len(total_df))
    # add to the file
        
    with open(save_path, "a") as f:
        for i, row in total_df.iterrows():
            f.write(json.dumps({"question": row["question"], "A": row["A"], "B": row["B"], "C": row["C"], "D": row["D"], "correct_answer": row["correct_answer"]}) + "\n")
    return total_df

def main():
    # read tasks from mmlu_tasks.jsnol
    with open(task_path, "r") as f:
        tasks = [json.loads(line) for line in f]
    print("Number of tasks: ", len(tasks))
    # read questions from full_questions.jsonl
    with open(full_questions_path, "r") as f:
        data = [json.loads(line) for line in f]
        print("Number of questions: ", len(data))
    # save answers for each task
    count = 0
    with open(save_path, "w") as f:
        for task in tasks:
            for d in data:
                if d["question"] in task["task"]:
                    task["answer"] = d["correct_answer"]
                    f.write(json.dumps(task) + "\n")
                    count += 1
                    break
        print("Number of tasks with answers: ", count)
main()
            

    
        