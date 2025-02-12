from metagpt.ext.aflow.benchmark.benchmark import BaseBenchmark
from metagpt.logs import logger
import glob
import pandas as pd
from typing import Union, List, Literal, Any, Dict, Callable,  Tuple
import re
import string
from collections import Counter

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed
import numpy as np
from abc import ABC
from experiments.scripts.utils.extract_MMLU_workflow import CallGraphParser
from datetime import datetime
from experiments.scripts.pgy_dataset import CustomGraphDataset
from torch_geometric.loader import DataLoader
import torch



class MMLUBenchmark(BaseBenchmark):
    def __init__(self, name: str, file_path: str, log_path: str):
        super().__init__(name, file_path, log_path)
        self.workflows_file = None

    def postprocess_answer(self, answer: Union[str, List[str]]) -> str:
        if isinstance(answer, list):
            if len(answer) > 0:
                answer = answer[0]
            else:
                answer = ""
        if not isinstance(answer, str):
            raise Exception("Expected string")
        if len(answer) == 1:
            return answer
        if answer.startswith("Option"):
            return answer[6]
        if len(answer) > 0:
            ans_pos = answer.find("answer is")
            if ans_pos != -1:
                extracted = answer[ans_pos + len("answer is"):].strip(":").strip().strip("*").strip().strip("Option").strip()
                match = re.search(r'([A-Z])', extracted, re.IGNORECASE)
                if match:
                    return match.group(1)
                answer = answer[ans_pos+len("answer is"):].strip(":").strip().strip("*").strip().strip("Option").strip()
            answer = answer[0] # Try to format the answer by taking the first letter
        return answer
    
    
    def calculate_score(self, expected_output: str, prediction: str) -> Tuple[float, str]:
        prediction = self.postprocess_answer(prediction)
        if prediction == expected_output:
            return 1.0, prediction
        else:
            return 0.0, prediction
        
    
    @retry(stop=stop_after_attempt(5), wait=wait_fixed(1), retry=retry_if_exception_type(Exception), reraise=True)
    async def _generate_output(self, graph, input_text):
        return await graph(input_text)
    
    async def evaluate_problem(self, problem: dict, graph: Callable) -> Tuple[str, str]:
        input_text = problem["task"] 
        expected_output = problem["answer"]
        
        try:
            output, cost = await self._generate_output(graph, input_text)
            score, extracted_output = self.calculate_score(expected_output, output)
            self.log_mismatch(input_text, expected_output, output, extracted_output)
            
            return input_text, output, extracted_output, expected_output, score, cost

        except Exception as e:
            logger.info(f"Maximum retries reached. Skipping this sample. Error: {e}")
            print(e)
            return input_text, str(e), str(e), expected_output, 0.0, 0.0
        
    async def gnn_evaluate_problem(self, data: List[dict], graph: Callable) -> Tuple[str, str]:
        # unstable extraction
        nodes, edges = self.extract_workflow(graph)

        raw_data_list = []
        for problem in data:
            item = {
                "task":problem["task"],
                "nodes": nodes,
                "edge_index": edges
            }
            raw_data_list.append(item)
        
        # set model and device in benchmark
        model = self.model
        device = self.device
        encoder = self.encoder
        
        

        raw_data_list = self.process_raw_data_list(raw_data_list)
        dataset = CustomGraphDataset(raw_data_list = raw_data_list, encoder = encoder)
        loader = DataLoader(dataset.data_list, batch_size=len(data), shuffle=False)

        
        try:
            scores = []
            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(device)
                    num_graphs = batch.batch[-1] + 1    
                    batch.task_embedding = batch.task_embedding.reshape(num_graphs, -1)
                    score = model(batch.x, batch.task_embedding, batch.edge_index, batch.batch)
                    preds = (score > 0.5).float()
                    scores.append(preds)
            
            output = []
            for i, problem in enumerate(data):
                update_item =[
                    problem["task"], "", "", problem["answer"], score, 0
                ]
            
            # return batch's output
            return output
            # self.log_mismatch(input_text, expected_output, output, extracted_output)
            # return input_text, output, extracted_output, expected_output, score, cost

        except Exception as e:
            logger.info(f"Maximum retries reached. Skipping this sample. Error: {e}")
            print(e)
            return "", str(e), str(e), "", 0.0, 0.0
    
    def get_result_columns(self) -> List[str]:
        return ["input_text", "output_text", "prediction", "answer", "score", "cost"]
    
    def extract_workflow(self, graph):
        # original: extarct from flexible codes
        pass

    def process_raw_data_list(self, raw_data_list, output_data_list):
        output_data_list = []
        for item in raw_data_list:
            item['source'] = 'mmlu_aflow'   
            item['model'] = 'gpt-4o-mini'   
            node_info = {}
            item['nodes'][0]['prompt'] = "This workflow starts with task initialization"
            for node in item['nodes']:
                node_info[node['name']] = node['prompt']
            node_id_map = {k:i for i,k in enumerate(node_info.keys())}  
            temp_node_info = {node_id_map[k]:v for k,v in node_info.items()}
            item['nodes'] = temp_node_info
            temp_edge_index = [[node_id_map[e['input']],node_id_map[e['output']]] for e in item['edges']]    
            item['edge_index'] = temp_edge_index
            keys_to_keep = ['nodes','edge_index','task','source','model']  
            item = {k:item[k] for k in keys_to_keep}    
            output_data_list.append(item)  
        
        return output_data_list