import os
import json

results_dir = "/home/ubuntu/DATA2/yuchenhou/GNN/MetaGPT/metagpt/ext/aflow/scripts/optimized/HotpotQA/results"
save_dir = "/home/ubuntu/DATA2/yuchenhou/GNN/MetaGPT/experiments/MMLU_"
new_results_dir = "/home/ubuntu/DATA2/yuchenhou/GNN/MetaGPT/updated/HotpotQA/results"

def main():
        
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    num_results = 0


    for subdir in os.listdir(results_dir):
        if os.path.isdir(os.path.join(results_dir, subdir)):
            workflow_id  = subdir
            for file in os.listdir(os.path.join(results_dir, subdir)):
                if  "log" in file:
                    continue
                score = file.split("_")[0]
                if float(score) < 0.1:
                    continue
                result_dir = os.path.join(results_dir, subdir)
                # copy result_dir to save_dir
                
                num_results += 1
                save_name = os.path.join(save_dir, subdir)
                os.system("cp -r {} {}".format(result_dir, save_name))
                print("Workflow {} has been chosen.".format(workflow_id))
                # if num_results >= 10:
                #     return
                break
    print("Total number of chosen results: {}".format(num_results))
            
def copy_new_results():
    num_results = 0
    for subdir in os.listdir(new_results_dir):
        if os.path.isdir(os.path.join(new_results_dir, subdir)):
            workflow_id  = subdir.split("_")[-1]
            new_workflow_id = int(workflow_id) +100
            new_dir_name = "round_{}".format(new_workflow_id)
            os.system("cp -r {} {}".format(os.path.join(new_results_dir, subdir), os.path.join(save_dir, new_dir_name)) )
    
main()    
copy_new_results()     
            
