from typing import Literal
import metagpt.ext.aflow.scripts.optimized.GSM8K.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.GSM8K.workflows.round_86.prompt as prompt_custom
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
        # Clarify problem requirements using self-ask method 
        clarification_response = await self.custom(input=problem, instruction="What are the essential details I should know to solve this problem?") 

        # Generate initial and alternative solutions using the custom operator, incorporating clarifications.
        initial_solution = await self.custom(input=clarification_response['response'] + problem, instruction="Can you help me solve it?") 
        additional_solution = await self.custom(input=clarification_response['response'] + problem, instruction="Can you provide another way to find the solution?") 
        
        # Introduce a review phase to evaluate solutions.
        review = f"Review the solutions: {initial_solution['response']} and {additional_solution['response']}. Please analyze their strengths, weaknesses, and provide a comparison."
        review_response = await self.custom(input=review, instruction="") 

        # Validation phase: Use self-consistency to validate and select the best solution. 
        ensemble_response = await self.sc_ensemble(solutions=[initial_solution['response'], additional_solution['response']], problem=problem) 
        
        # Provide analysis with the programmer for better clarity of solutions and their validation. 
        analysis_response = await self.programmer(problem=problem, analysis=f"Initial Solution: {initial_solution['response']}, Additional Solution: {additional_solution['response']}, Review: {review_response['response']}, Selected Solution: {ensemble_response['response']}") 
        
        return analysis_response['output'], self.llm.cost_manager.total_cost
