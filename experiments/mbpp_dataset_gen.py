import sys
import os
import argparse
import yaml
import json
import time
from tqdm import tqdm
import random
import asyncio
from pathlib import Path
import torch
import copy
from typing import List,Union,Literal
import random
import gc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append("/home/ubuntu/DATA2/yuchenhou/GNN/GDesigner-3063/")
sys.stdout.reconfigure(encoding='utf-8')

from my_datasets.gsm8k_dataset import gsm_get_predict

from GDesigner.graph.sim_workflow import SimWorkflow
from GDesigner.tools.reader.readers import JSONLReader
from GDesigner.tools.coding.python_executor import PyExecutor
from GDesigner.utils.globals import Time
from GDesigner.utils.const import GDesigner_ROOT
from GDesigner.utils.globals import Cost, PromptTokens, CompletionTokens




def load_result(result_file):
    if not result_file.exists():
        with open(result_file, 'w', encoding='utf-8') as file:
            json.dump([], file)

    with open(result_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def dataloader(data_list, batch_size, i_batch):
    return data_list[i_batch*batch_size:i_batch*batch_size + batch_size]


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def parse_args():
    parser = argparse.ArgumentParser(
        description="GDesigner Experiments on HumanEval")
    parser.add_argument("--dataset_json", type=str,
                        default=f"{GDesigner_ROOT}/my_datasets/mbpp.jsonl")
    parser.add_argument("--llm_name", type=str, default="gpt-4o-mini")
    parser.add_argument('--batch_size', type=int, default=4, help="batch size")
    parser.add_argument('--num_rounds', type=int, default=2,
                        help="Number of optimization/inference rounds for one query")
    parser.add_argument('--pruning_rate', type=float,
                        default=0.25, help="The Rate of Pruning. Default 0.05.")
    parser.add_argument('--domain', type=str, default="mbpp",
                        help="Domain (the same as dataset name), default 'humaneval'")
    parser.add_argument('--chosen_workflow_file', type=str, default=f"{GDesigner_ROOT}/results/mbpp/edge_index/gpt-4o-mini_2025-01-17-00-45-23.json")
    parser.add_argument('--basedir', type=str, default=f"{GDesigner_ROOT}/results/")
    parser.add_argument('--decision_method', type=str, default="FinalWriteCode")
    parser.add_argument('--agent_name', nargs='+', type=str, default='CodeWriting')
    parser.add_argument('--workflow_id', type=int, default=0,)
    parser.add_argument('--workflow_batch', type=int, default=0,)
    parser.add_argument('--workflow_batch_size', type=int, default=10,)
    args = parser.parse_args()
    return args


async def main():
    args = parse_args()
    import random
    chosen_workflows_file = args.chosen_workflow_file
    

    workflow_data = load_result(Path(chosen_workflows_file))
    workflow_data = workflow_data[args.workflow_batch_size*args.workflow_batch:args.workflow_batch_size*args.workflow_batch + args.workflow_batch_size]
    print(f"Number of chosen workflows: {len(workflow_data)}")
    dataset = JSONLReader.parse_file(args.dataset_json)
    domain = args.domain
    current_time = Time.instance().value or time.strftime(
        "%Y-%m-%d-%H-%M-%S", time.localtime())
    Time.instance().value = current_time
    workflow_file_name = chosen_workflows_file.split('/')[-1].split('.')[0]
    result_dir = Path(f"{args.basedir}/{domain}/new_workflow-label_pairs/{workflow_file_name}/")
    result_dir.mkdir(parents=True, exist_ok=True)
    decision_method = args.decision_method
    
    for i, workflow in enumerate(workflow_data):
        workflow_id = args.workflow_batch_size*args.workflow_batch + i
        print(f"Workflow {i}")
        total_executed = 0
        total_solved = 0

        result_file = result_dir / f"{args.llm_name}_workflow_{workflow_id}_dataset.json"
        if result_file.exists():
            resume = True
            resume_data = load_result(result_file)
            if len(resume_data) == len(dataset):
                continue
            else: 
                start = int(len(resume_data)/args.batch_size)
        else:
            resume = False
            start = 0
        ori_simworkflow = SimWorkflow(workflow= workflow,
                            domain=domain,
                            llm_name=args.llm_name,
                            decision_method=decision_method,
                            agent_name= args.agent_name)
        edge_index = ori_simworkflow.spatial_edge_index()
        nodes_info = {ori_simworkflow.nodes[node].id: ori_simworkflow.nodes[node].role for node in ori_simworkflow.nodes}
        num_batches = int(len(dataset)//args.batch_size) + 1
        batches = range(start,num_batches)
        
        print(f"Total batches: {num_batches}")
        ####################################
        for i_batch in tqdm(batches,desc="loop of batches"):
            answers = []
            print(f"Batch {i_batch}", 80*'-')
            tests = []

            current_batch = dataloader(dataset, args.batch_size, i_batch)
            if current_batch is None:
                print("No more data available.")
                break
            for _, record in enumerate(current_batch):
                simworkflow = copy.deepcopy(ori_simworkflow)
                task = record["prompt"]
                test = record["test"]
                test_imports = record["test_imports"]
                if test_imports:
                    for i in range(len(test_imports)):
                        test = test_imports[i] + test
                tests.append(test)
                input_dict = {"task": task}
                answers.append(asyncio.create_task(
                    simworkflow.arun(input_dict)))
                del simworkflow
            raw_results = await asyncio.gather(*answers)
            raw_answers = raw_results
            gc.collect()
            utilities: List[float] = []
            data = load_result(result_file)

            # print(edge_index)
            for task, answer, test in zip(current_batch, raw_answers, tests):
                
                if not isinstance(answer, list):
                    raise TypeError(
                        f"Expected a list for the answer, but got {type(answer).__name__}")
                answer = answer[0].lstrip("```python\n").rstrip("\n```")
                print(f"Task: {task}")
                print(f"Answer: {answer}")
                is_solved, _, _ = PyExecutor().execute(answer, [test], timeout=100)
                print(f"Solved: {is_solved}")
                total_solved = total_solved + is_solved
                total_executed = total_executed + 1
                accuracy = total_solved / total_executed
                utility = is_solved
                utilities.append(utility)
                updated_item = {
                    "task": task,
                    "tests": test,
                    "prediction": answer,
                    "score": is_solved,
                    "nodes": nodes_info,
                    "edge_index": edge_index,
                    "Total executed": total_executed,
                    "Accuracy": accuracy
                }
                data.append(updated_item)
                print(f"Utility: {utility}")
                print(f"Total solved: {total_solved}")
                print(f"Total executed: {total_executed}")
                print(f"Accuracy: {accuracy}")
                print(f"Cost {Cost.instance().value}")
                
            with open(result_file, 'w', encoding='utf-8') as file:
                json.dump(data, file, indent=4)
            # if total_executed ==20:
            #     break
            print(f"Cost {Cost.instance().value}")
            print(f"PromptTokens {PromptTokens.instance().value}")
            print(f"CompletionTokens {CompletionTokens.instance().value}")




if __name__ == '__main__':
    asyncio.run(main())
