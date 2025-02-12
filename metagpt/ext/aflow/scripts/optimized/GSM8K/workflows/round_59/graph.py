from typing import Literal
import metagpt.ext.aflow.scripts.optimized.GSM8K.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.GSM8K.workflows.round_59.prompt as prompt_custom
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
        # Generate problem analysis before finding solutions
        problem_analysis = await self.custom(input=f"Analyze the following problem for clarity: {problem}", instruction="Please provide insights on how to approach this problem.") 
        
        # Generate initial and alternative solutions using the custom operator based on the analysis.
        initial_solution = await self.custom(input=f"Based on the analysis, solve this problem: {problem}.", instruction="") 
        additional_solution = await self.custom(input=f"Based on the analysis, can you offer another solution to the problem: {problem}?", instruction="") 
        
        # Validation phase: Use self-consistency to validate and select the best solution. 
        ensemble_response = await self.sc_ensemble(solutions=[initial_solution['response'], additional_solution['response']], problem=problem) 
        
        # Provide analysis with the programmer for better clarity of solutions and their validation. 
        analysis_response = await self.programmer(problem=problem, analysis=f"Problem Analysis: {problem_analysis['response']}; Initial Solution: {initial_solution['response']}; Additional Solution: {additional_solution['response']}; Selected Solution: {ensemble_response['response']}") 
        
        return analysis_response['output'], self.llm.cost_manager.total_cost
