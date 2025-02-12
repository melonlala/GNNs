from typing import Literal
import metagpt.ext.aflow.scripts.optimized.GSM8K.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.GSM8K.workflows.round_74.prompt as prompt_custom
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

    async def __call__(self, problem: str): 
        """ 
        Implementation of the workflow 
        """ 
        # Generate multiple solutions using the custom operator. 
        solutions = await self.custom(input=f"Please solve the following problem: {problem}", instruction="Provide two distinct solutions.") 
        
        # Review the generated solutions
        reviewed_solutions = [solution for solution in solutions['response'].split(';') if solution.strip()]
        
        # Use self-consistency to select the best solution. 
        ensemble_response = await self.sc_ensemble(solutions=reviewed_solutions, problem=problem) 
        
        # Provide analysis with the programmer for better clarity of solutions and their validation. 
        analysis_response = await self.programmer(problem=problem, analysis=f"Solutions: {', '.join(reviewed_solutions)}, Selected Solution: {ensemble_response['response']}") 
        
        return analysis_response['output'], self.llm.cost_manager.total_cost
