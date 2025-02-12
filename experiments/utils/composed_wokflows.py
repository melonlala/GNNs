import os
import json
base_dir = "./simulations/eval/full_humaneval"
save_path = "./large_scale_workflow/composed_workflows/full_composed_dataset.jsonl"
dataset_path = "./datasets/humaneval/humaneval-py.jsonl"
workflow_path = "/home/ubuntu/DATA2/yuchenhou/GNN/GDesigner-3063/large_scale_workflow/composed_workflows/humaneval/chosen_workflows.json"
def get_data(item):
    print(item)
    id_name = item['Question']['name']
    prompt = item['Question']['prompt']
    test = item['Tests']
    is_solved = item['Solved']
    output = {
        'id': id_name,
        'prompt': prompt,
        'test': test,
        'label': is_solved,
        'task': item['Question']
    }
    return output


def get_workflow(workflow_id):
    with open(workflow_path, 'r') as f:
        data = json.load(f)
        workflow = data[workflow_id]
        nodes = workflow['nodes']
        edge_index = workflow['edge_index']
        output = {
            'nodes': nodes,
            'edge_index': edge_index
        }
        return output
        
        
# count = 0
composed_data = []
for file in os.listdir(base_dir):
    workflow_id = int(file.split(".")[0].split("_")[-1])
    if not file.endswith(".json"):
        continue
    with open(os.path.join(base_dir, file), 'r') as f:
        data = json.load(f)
        # task_id = count
        for item in data:
            label_output = get_data(item)
            workflow_output = get_workflow(workflow_id)
            output = {
                'id': label_output['id'],
                'label': label_output['label'],
                'prompt': label_output['prompt'],
                'test': label_output['test'],
                'nodes': workflow_output['nodes'],
                'edge_index': workflow_output['edge_index'],
                'task': label_output['task']
            }
            composed_data.append(output)
    
    print(f"len(composed_data): {len(composed_data)}")
    with open(save_path, 'w') as f:
        for item in composed_data:
            json.dump(item, f)
            f.write('\n')
        
            
        

