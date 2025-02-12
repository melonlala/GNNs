import os
import json
import random
label_dir = './simulations/eval/results/full_humaneval/'
workflow_file = './large_scale_workflow/composed_workflows/humaneval/chosen_workflows.json'
save_file = './large_scale_workflow/composed_workflows/full_composed_dataset.jsonl'

def load_results(file_path):
    with open(file_path, 'r') as f:
        results = json.load(f)
    return results

workflow_data = load_results(workflow_file)

dataset = []
count_t = 0
count_f = 0
avg_solved_rate = []
for i, workflow in enumerate(workflow_data):
    workflow_id = 'workflow_'+str(i)
    lable_file = os.path.join(label_dir, workflow_id+'.json')
    if not os.path.exists(lable_file):
        continue
    label_data = load_results(lable_file)
    nodes = workflow['nodes']
    edge_index = workflow['edge_index']
    if len(label_data)!= 31:
        continue
    temp_t = 0
    temp_f = 0
    for j, record in enumerate(label_data):
        
        label = record['Solved']
        if label == True:
            count_t += 1
            temp_t += 1
        else:
            count_f += 1
            temp_f += 1
        task_name = record['Question']['name']
        prompt = record['Question']['prompt']
        test = record['Tests']
        task = record['Question']
        data = {
            'id': task_name,
            'label': label,
            'prompt': prompt,
            'test': test,
            'nodes': nodes,
            'edge_index': edge_index,
            'task': task
        }
        dataset.append(data)
    avg_solved_rate.append(temp_t/(temp_t+temp_f))

with open(save_file, 'w') as f:
    for data in dataset:
        json.dump(data, f)
        f.write('\n')
            

print('Average solved rate:', sum(avg_solved_rate)/len(avg_solved_rate))
print('Dataset size:', len(dataset))
print('True:', count_t)
print('False:', count_f)

        