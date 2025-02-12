from typing import Literal
import metagpt.ext.aflow.scripts.optimized.MBPP.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.MBPP.workflows.round_37.prompt as prompt_custom
from metagpt.provider.llm_provider_registry import create_llm_instance
from metagpt.utils.cost_manager import CostManager

DatasetType = Literal["HumanEval", "MBPP", "GSM8K", "MATH", "HotpotQA", "DROP", "MMLU"]

class Workflow: 
    def __init__(self, 
        name: str, 
        llm_config, 
        dataset: DatasetType, 
    ) -> None: 
        self.name = name 
        self.dataset = dataset 
        self.llm = create_llm_instance(llm_config) 
        self.llm.cost_manager = CostManager() 
        self.custom = operator.Custom(self.llm) 
        self.custom_code_generate = operator.CustomCodeGenerate(self.llm) 
        self.test_operator = operator.Test(self.llm) 
        self.sc_ensemble = operator.ScEnsemble(self.llm) 

    async def __call__(self, problem: str, entry_point: str): 
        solutions = [] 
        attempts = 0  # Initialize attempts counter
        while len(solutions) < 3 and attempts < 5:  # Allow up to 5 attempts to generate solutions
            solution = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction="") 
            print("Generated solution:", solution['response']) 
            if self.validate_solution(solution['response']):  # Validate before adding
                solutions.append(solution['response']) 
            attempts += 1
        
        # Provide feedback on how many attempts were made
        print(f"Total attempts to generate valid solutions: {attempts}")
        
        # Use ScEnsemble to select the best solution from the valid solutions
        best_solution = await self.sc_ensemble(solutions=solutions, problem=problem) 
        
        test_result = await self.test_operator(problem=problem, solution=best_solution['response'], entry_point=entry_point) 
        
        if not test_result['result']: 
            print("Test failed. Current solution:", best_solution['response']) 
            
        return best_solution['response'], self.llm.cost_manager.total_cost 

    def validate_solution(self, solution: str) -> bool: 
        return bool(solution)  # Improved checking logic could be added here
