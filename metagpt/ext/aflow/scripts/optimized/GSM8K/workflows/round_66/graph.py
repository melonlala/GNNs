from typing import Literal
import metagpt.ext.aflow.scripts.optimized.GSM8K.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.GSM8K.workflows.round_66.prompt as prompt_custom
from metagpt.provider.llm_provider_registry import create_llm_instance
from metagpt.utils.cost_manager import CostManager

DatasetType = Literal["HumanEval", "MBPP", "GSM8K", "MATH", "HotpotQA", "DROP", "MMLU"]

class Workflow: 
    def __init__(self, name: str, llm_config, dataset: DatasetType) -> None: 
        self.name = name 
        self.dataset = dataset 
        self.llm = create_llm_instance(llm_config) 
        self.llm.cost_manager = CostManager() 
        self.custom = operator.Custom(self.llm) 
        self.programmer = operator.Programmer(self.llm) 
        self.sc_ensemble = operator.ScEnsemble(self.llm) 
        self.self_ask = operator.SelfAsk(self.llm)  # Introduced selfAsk operator

    async def __call__(self, problem: str): 
        """ 
        Implementation of the workflow 
        """ 
        # Generate initial and alternative solutions using the custom operator. 
        initial_solution = await self.custom(input=f"Here is the problem: {problem}. Can you help me solve it?", instruction="") 
        additional_solution = await self.custom(input=f"Here is the problem: {problem}. Can you provide another way to find the solution?", instruction="") 
        
        # Validation phase: Use self-consistency to validate and select the best solution. 
        ensemble_response = await self.sc_ensemble(solutions=[initial_solution['response'], additional_solution['response']], problem=problem) 
        
        # Incorporate self-asking mechanism for clarity in the analysis phase
        self_ask_response = await self.self_ask(question=problem, response=ensemble_response['response'])
        
        # Provide thorough analysis with the programmer for clarity of solutions and their validation. 
        analysis_response = await self.programmer(problem=problem, analysis=f"Initial Solution: {initial_solution['response']}, Additional Solution: {additional_solution['response']}, Selected Solution: {self_ask_response['response']}") 
        
        return analysis_response['output'], self.llm.cost_manager.total_cost
