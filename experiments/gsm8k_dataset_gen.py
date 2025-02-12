import sys
import os
import argparse
import yaml
import json
import time
import asyncio
from pathlib import Path
import torch
import copy
from typing import List,Union,Literal
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')
os.environ["HTTP_PROXY"] = "127.0.0.1:7835"
os.environ["HTTPS_PROXY"] = "127.0.0.1:7835"

from GDesigner.agents.profile import get_profile
from GDesigner.utils.const import GDesigner_ROOT
from GDesigner.graph.graph import Graph
from GDesigner.graph.sim_workflow import SimWorkflow
from GDesigner.tools.reader.readers import JSONLReader
from GDesigner.utils.globals import Time
from GDesigner.utils.globals import Cost, PromptTokens, CompletionTokens
from my_datasets.gsm8k_dataset import gsm_data_process,gsm_get_predict
import os
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7834"
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7834"
def load_result(result_file):
    if not result_file.exists():
        with open(result_file, 'w',encoding='utf-8') as file:
            json.dump([], file)

    with open(result_file, 'r',encoding='utf-8') as file:
        data = json.load(file)
    return data

def dataloader(data_list, batch_size, i_batch):
    return data_list[i_batch*batch_size:i_batch*batch_size + batch_size]

def load_config(config_path):
    with open(config_path, 'r',encoding='utf-8') as file:
        return yaml.safe_load(file)
    
def parse_args():
    parser = argparse.ArgumentParser(description="GDesigner Experiments on gsm8k")
    parser.add_argument("--dataset_json", type=str, default="datasets/gsm8k/gsm8k.jsonl")
    parser.add_argument("--result_file", type=str, default=None)
    parser.add_argument("--llm_name", type=str, default="gpt-4o")
    parser.add_argument('--mode', type=str, default='FullConnected',
                        choices=['DirectAnswer', 'FullConnected', 'Random', 'Chain','Debate','Layered','Star'],
                        help="Mode of operation. Default is 'FullConnected'.")
    parser.add_argument('--lr', type=float, default=0.1,help="learning rate")
    parser.add_argument('--batch_size', type=int, default=16,help="batch size")
    parser.add_argument('--num_rounds',type=int,default=1,help="Number of optimization/inference rounds for one query")
    parser.add_argument('--pruning_rate', type=float, default=0.25,help="The Rate of Pruning. Default 0.05.")
    parser.add_argument('--num_iterations', type=int, default=10,help="The num of training iterations.")
    parser.add_argument('--domain', type=str, default="gsm8k",help="Domain (the same as dataset name), default 'gsm8k'")
    parser.add_argument('--agent_names', nargs='+', type=str, default=['MathSolver'],
                        help='Specify agent names as a list of strings')
    parser.add_argument('--agent_nums', nargs='+', type=int, default=[4],
                        help='Specify the number of agents for each name in agent_names')
    parser.add_argument('--decision_method', type=str, default='FinalRefer',
                        help='The decison method of the GDesigner')
    parser.add_argument('--optimized_spatial', default=True)
    parser.add_argument('--optimized_temporal',default=True)
    parser.add_argument('--profile_choice', type=int, default=0,help="The choice of the profile")
    args = parser.parse_args()
    result_path = GDesigner_ROOT / "result"
    os.makedirs(result_path, exist_ok=True)

    return args

async def main():
    args = parse_args()
    result_file = None
    chosen_workflows_file = Path(f"{GDesigner_ROOT}/large_scale_workflow/graphs/gsm8k/chosen/num_6_profile_0_FullConnected_gpt-4o_2024-11-20-17-39-55.json")
    chosen_workflows_file_id = chosen_workflows_file.stem.split('/')[-1].split('.')[0]
    workflow_data = load_result(chosen_workflows_file)
    workflow_data = workflow_data# only use the first 10 workflows for testing
    print(f"Number of chosen workflows: {len(workflow_data)}")

    dataset = JSONLReader.parse_file(args.dataset_json)
    dataset = gsm_data_process(dataset)
    import random
    dataset = random.sample(dataset, len(dataset)) # shuffle the dataset

    current_time = Time.instance().value or time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    Time.instance().value = current_time
    result_dir = Path(f"{GDesigner_ROOT}/simulations/eval/{args.domain}/{chosen_workflows_file_id}")
    result_dir.mkdir(parents=True, exist_ok=True)
    
    decision_method = args.decision_method
    for i, workflow in enumerate(workflow_data):
        # fix agent_names
        workflow["agent_names"] = args.agent_names * len(workflow["nodes"])
        print(f"Workflow {i}-------------------------------------------------------------------")

        result_file = result_dir / f"{args.llm_name}_{current_time}_workflow_{i}.json"
        simworkflow = SimWorkflow(workflow= workflow,
                            domain="gsm8k",
                            llm_name=args.llm_name,
                            decision_method=decision_method)
        
        
        # dataset = [record for record in dataset if record not in done_data["Question"]]
        num_batches = int(len(dataset)//args.batch_size)+1
        print(f"Number of batches: {num_batches}")
        total_solved, total_executed = (0, 0)
        
        for i_batch in range(num_batches):
            print(f"Batch {i_batch}",80*'-')
            start_ts = time.time()
            answer_log_probs = []
            answers = []
            
            current_batch = dataloader(dataset,args.batch_size,i_batch)
            if current_batch is None:
                print("No more data available.")
                break
            
            for i_record, record in enumerate(current_batch):
                task = record["task"]
                step = record["step"]
                answer = record["answer"]
                answers.append(answer)
                input_dict = {"task": task}
                answer_log_probs.append( simworkflow.arun(input_dict))
            raw_results = await asyncio.gather(*answer_log_probs)
            raw_answers = raw_results
            
            utilities: List[float] = []
            data = load_result(result_file)
            
            for task, answer, true_answer in zip(current_batch, raw_answers, answers):
                
                predict_answer = gsm_get_predict(answer[0])
                is_solved = float(predict_answer)==float(true_answer)
                total_solved = total_solved + is_solved
                total_executed = total_executed + 1
                accuracy = total_solved/ total_executed
                utility = is_solved
                utilities.append(utility)
                
                updated_item = {
                    "Question": task,
                    "Answer": true_answer,
                    "Step": step,
                    "Response": answer,
                    "Attempt answer": predict_answer,
                    "Solved": is_solved,
                    "Total solved": total_solved,
                    "Total executed": total_executed,
                    "Accuracy": accuracy
                }
                
                data.append(updated_item)
                print(f"Total solved: {total_solved}")
                print(f"Total executed: {total_executed}")
                print(f"Batch time {time.time() - start_ts:.3f}")
                print(f"Accuracy: {accuracy}")
                print("utilities:", utilities)
        
            with open(result_file, 'w',encoding='utf-8') as file:
                json.dump(data, file, indent=4)
            
            print(f"Cost {Cost.instance().value}")
            print(f"PromptTokens {PromptTokens.instance().value}")
            print(f"CompletionTokens {CompletionTokens.instance().value}")
 

if __name__ == '__main__':
    asyncio.run(main())
