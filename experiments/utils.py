import os
import json
def compose_dataset(data_dir,result_dir):
    data = []
    for file in os.listdir(data_dir):
        if not os.path.exists(result_dir):
            os.makedirs(result_dir) 
        if file.endswith(".json"):
            with open(os.path.join(data_dir, file), "r") as f:
                data += json.load(f)
                
    with open(os.path.join(result_dir, "workflows.json"), "w",encoding='utf-8') as f: 
        json.dump(data, f,indent=4)
    return data


if __name__ == "__main__":
    data_dir = "./large_scale_workflow/graphs/gsm8k/chosen"
    result_dir = "./large_scale_workflow/composed_workflows/gsm8k"
    data = compose_dataset(data_dir,result_dir)
    print(len(data))
                

