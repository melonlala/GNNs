from typing import Literal
import metagpt.ext.aflow.scripts.optimized.GSM8K.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.GSM8K.workflows.round_47.prompt as prompt_custom
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
        # Generate initial and alternative solutions using the custom operator. 
        initial_solution = await self.custom(input=f"Here is the problem: {problem}. Can you help me solve it comprehensively?", instruction="") 
        additional_solution = await self.custom(input=f"Here is the problem: {problem}. Can you provide another solution approach?", instruction="") 
        
        # Ensure responses are non-empty and provide defaults if necessary.
        initial_response = initial_solution['response'] if initial_solution['response'] else "No response generated."
        additional_response = additional_solution['response'] if additional_solution['response'] else "No response generated."
        
        # Review phase: integrate both solutions.
        review_response = await self.custom(input=f"Initial Solution: {initial_response}. Additional Solution: {additional_response}. Review both and summarize key differences.", instruction="") 
        
        # Validation phase: Use self-consistency to validate and select the best solution. 
        ensemble_response = await self.sc_ensemble(solutions=[initial_response, additional_response], problem=problem) 
        
        # Provide analysis with the programmer for clarity on solutions, validations, and reviews.
        analysis_response = await self.programmer(problem=problem, analysis=f"Review: {review_response['response']}. Initial Solution: {initial_response}. Additional Solution: {additional_response}. Selected Solution: {ensemble_response['response']}") 
        
        return analysis_response['output'], self.llm.cost_manager.total_cost
