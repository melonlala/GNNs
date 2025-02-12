from typing import Literal
import metagpt.ext.aflow.scripts.optimized.MATH.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.MATH.workflows.round_140.prompt as prompt_custom
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
        self.ensemble = operator.ScEnsemble(self.llm)  

    async def __call__(self, problem: str):  
        # Generate initial solution  
        solution = await self.custom(input=problem, instruction=prompt_custom.SOLVE_PROMPT)  
        
        # Revise the solution based on preliminary review  
        revise_instruction = f"Revise the following solution for potential improvements: {solution['response']}"  
        revised_solution = await self.custom(input=revise_instruction, instruction=prompt_custom.REVISE_PROMPT)  

        # Review the revised solution for correctness  
        review_instruction = f"Review the following solution for correctness: {revised_solution['response']}"  
        review_result = await self.custom(input=review_instruction, instruction=prompt_custom.REVIEW_PROMPT)  
        
        # Validate the solution using Python code  
        validation_code = f"def validate_solution():\n    # Your validation logic here\n    return {review_result['response']}\n"  
        validation_results = await self.programmer(problem=validation_code)  

        # Multi-Validate by running several validation checks  
        multi_validation_code = f"def multi_validate_solution():\n    # Your multiple validation logic here\n    return [{validation_results['output']}, ...]\n"  
        multi_validation_results = await self.programmer(problem=multi_validation_code)  

        # Use ensemble to improve the final answer selection  
        ensemble_result = await self.ensemble(solutions=[revised_solution['response'], multi_validation_results['output']], problem=problem)  
        
        # Return the final solution and cost  
        return ensemble_result['response'], self.llm.cost_manager.total_cost
