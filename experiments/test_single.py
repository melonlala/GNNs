import sys
import os
import argparse
import yaml
import json
import time
import random
import asyncio
from pathlib import Path
import torch
import copy
from typing import List,Union,Literal
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

from GDesigner.graph.graph import Graph
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
                        default="datasets/humaneval/filter_results_2024-11-14-14-59-42/all_false_tasks.jsonl")
    parser.add_argument("--result_file", type=str, default=None)
    parser.add_argument("--llm_name", type=str, default="gpt-4o-mini")
    parser.add_argument('--mode', type=str, default='FullConnected',
                        choices=['DirectAnswer', 'FullConnected',
                                 'Random', 'Chain', 'Debate', 'Layered', 'Star'],
                        help="Mode of operation. Default is 'FullConnected'.")
    parser.add_argument('--lr', type=float, default=0.1, help="learning rate")
    parser.add_argument('--batch_size', type=int, default=4, help="batch size")
    parser.add_argument('--num_rounds', type=int, default=2,
                        help="Number of optimization/inference rounds for one query")
    parser.add_argument('--pruning_rate', type=float,
                        default=0.25, help="The Rate of Pruning. Default 0.05.")
    parser.add_argument('--num_iterations', type=int,
                        default=50, help="The num of training iterations.")
    parser.add_argument('--domain', type=str, default="humaneval",
                        help="Domain (the same as dataset name), default 'humaneval'")
    parser.add_argument('--agent_names', nargs='+', type=str, default=['CodeWriting'],
                        help='Specify agent names as a list of strings')
    parser.add_argument('--agent_nums', nargs='+', type=int, default=[6],
                        help='Specify the number of agents for each name in agent_names')
    parser.add_argument('--decision_method', type=str, default='FinalWriteCode',
                        help='The decison method of the GDesigner')
    parser.add_argument('--optimized_spatial',type=bool,default=True)
    parser.add_argument('--optimized_temporal',type=bool,default=True)

    args = parser.parse_args()
    result_path = GDesigner_ROOT / "result"
    os.makedirs(result_path, exist_ok=True)
    if len(args.agent_names) != len(args.agent_nums):
        parser.error(
            "The number of agent names must match the number of agent counts.")

    return args


async def main():
    args = parse_args()

    # save the chosen workflows to a file
    # workflow_data_file = Path(f"{GDesigner_ROOT}/large_scale_workflow/composed_workflows/workflows.json")
    # workflow_data = load_result(workflow_data_file)
    # print(f"Number of workflows: {len(workflow_data)}")
    import random
    chosen_workflows_file = Path(f"{GDesigner_ROOT}/large_scale_workflow/composed_workflows/diy_workflows.json")
    workflow_data = load_result(chosen_workflows_file)
    workflow = workflow_data[0]
    print(f"Number of chosen workflows: {len(workflow_data)}")
    dataset = JSONLReader.parse_file(args.dataset_json)
    
    # random.shuffle(dataset)
    current_time = Time.instance().value or time.strftime(
        "%Y-%m-%d-%H-%M-%S", time.localtime())
    Time.instance().value = current_time
    result_dir = Path(f"{GDesigner_ROOT}/simulations/eval/single/")
    result_dir.mkdir(parents=True, exist_ok=True)
    # result_edge_index_dir = Path(f"{GDesigner_ROOT}/result/edge_index")
    # result_edge_index_dir.mkdir(parents=True, exist_ok=True)
    # result_edge_index = result_edge_index_dir/ f"{args.llm_name}_{current_time}.json"
    # agent_names = [name for name, num in zip(
    #     args.agent_names, args.agent_nums) for _ in range(num)]
    decision_method = args.decision_method
 

    total_executed = 0
    total_solved = 0

    result_file = result_dir / f"{args.llm_name}_{current_time}.json"
    simworkflow = SimWorkflow(workflow= workflow,
                        domain="humaneval",
                        llm_name=args.llm_name,
                        decision_method=decision_method)
    # edge_index = simworkflow.spatial_edge_index()
    
    num_batches = int(len(dataset)//args.batch_size) + 1
    print(f"Total batches: {num_batches}")
    ####################################
    for i_batch in range(num_batches):
        answers = []
        print(f"Batch {i_batch}", 80*'-')
        tests = []

        current_batch = dataloader(dataset, args.batch_size, i_batch)
        if current_batch is None:
            print("No more data available.")
            break
        for _, record in enumerate(current_batch):
            task = record["prompt"]
            test = record["test"]
            tests.append(test)
            input_dict = {"task": task}
            answers.append(asyncio.create_task(
                simworkflow.arun(input_dict)))
        raw_results = await asyncio.gather(*answers)
        raw_answers = raw_results
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
                "Question": task,
                "Tests": test,
                "Attempt answer": answer,
                "Solved": is_solved,
                "Solution": answer,
                "Total solved": total_solved,
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
