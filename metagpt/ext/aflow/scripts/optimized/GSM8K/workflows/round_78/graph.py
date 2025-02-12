from typing import Literal
import metagpt.ext.aflow.scripts.optimized.GSM8K.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.GSM8K.workflows.round_78.prompt as prompt_custom
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
        initial_solution = await self.custom(input=f"Here is the problem: {problem}. Can you help me solve it?", instruction="Please provide a detailed explanation of the solution process.") 
        additional_solution = await self.custom(input=f"Here is the problem: {problem}. Can you provide another way to find the solution?", instruction="Please provide a detailed explanation of the solution process.") 
        
        # Review phase: Check consistency and reasonableness of solutions prior to validation.
        reviewed_solutions = [initial_solution['response'], additional_solution['response']]
        if None in reviewed_solutions or '' in reviewed_solutions:
            raise ValueError("Solutions must not be None or empty.")

        # Validation phase: Validate generated solutions before analysis. 
        validated_solutions = reviewed_solutions
        
        # Use self-consistency to select the best solution. 
        ensemble_response = await self.sc_ensemble(solutions=validated_solutions, problem=problem) 
        
        # Provide analysis with the programmer for better clarity of solutions and their validation. 
        analysis_response = await self.programmer(problem=problem, analysis=f"Initial Solution: {initial_solution['response']}, Additional Solution: {additional_solution['response']}, Selected Solution: {ensemble_response['response']}") 
        
        return analysis_response['output'], self.llm.cost_manager.total_cost
