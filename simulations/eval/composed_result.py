import os
import json
import shutil

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

result_dir = '/home/ubuntu/DATA2/yuchenhou/GDesigner-3063/simulations/eval'
save_dir = '/home/ubuntu/DATA2/yuchenhou/GDesigner-3063/simulations/eval/results'
if os.path.exists(save_dir):
    shutil.rmtree(save_dir)
os.makedirs(save_dir, exist_ok=True)

# Collect unique file name IDs
file_names = {
    '_'.join(file.split('_')[-2:])
    for file in os.listdir(result_dir) if file.endswith('.json')
}
# Combine files based on file_name_id
for file_name_id in file_names:
    combined_results = {}  # Use a dictionary if files are dicts, list otherwise
    is_dict = False

    for file in os.listdir(result_dir):
        if file.endswith('.json') :
            tmp_file_name_id = '_'.join(file.split('_')[-2:])
            if tmp_file_name_id != file_name_id:
                continue
            file_path = os.path.join(result_dir, file)
            data = load_json(file_path)
            
            if isinstance(data, dict):
                is_dict = True
                combined_results.update(data)  # Merge dictionaries
            elif isinstance(data, list):
                if not isinstance(combined_results, list):
                    combined_results = []  # Initialize as list if data is list
                combined_results.extend(data)  # Merge lists
            # os.remove(file_path)  # Remove file after combining

    # Save combined results as a JSON file
    if combined_results:
        save_path = os.path.join(save_dir, file_name_id)
        save_json(combined_results, save_path)
        print(f"Saved combined file: {save_path}")
