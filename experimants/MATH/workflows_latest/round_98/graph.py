from typing import Literal
import metagpt.ext.aflow.scripts.optimized.MATH.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.MATH.workflows.round_98.prompt as prompt_custom
from metagpt.provider.llm_provider_registry import create_llm_instance
from metagpt.utils.cost_manager import CostManager

DatasetType = Literal["HumanEval", "MBPP", "GSM8K", "MATH", "HotpotQA", "DROP", "MMLU"]

class Workflow:
    def __init__(
        self,
        name: str,
        llm_config,
        dataset: DatasetType,
    ) -> None:
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.llm.cost_manager = CostManager()
        self.custom = operator.Custom(self.llm)
        self.programmer = operator.Programmer(self.llm)
        self.ensemble = operator.ScEnsemble(self.llm)

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        # Generate initial solution
        solution = await self.custom(input=problem, instruction=prompt_custom.SOLVE_PROMPT)
        
        # Revise and validate the initial solution
        revise_instruction = f"Revise the following solution for clarity and correctness: {solution['response']}"
        revised_solution = await self.custom(input=revise_instruction, instruction=prompt_custom.REVIEW_PROMPT)
        
        # Validate the solution using Python code
        validation_code = f"def validate_solution():\n    # Your validation logic here\n    return {revised_solution['response']}\n"
        validation_result = await self.programmer(problem=validation_code)
        
        # Additional review step to confirm correctness
        review_validation_instruction = f"Please confirm the following validation outcome: {validation_result['output']}"
        final_review_result = await self.custom(input=review_validation_instruction, instruction=prompt_custom.REVIEW_PROMPT)

        # Use ensemble to improve the final answer selection
        ensemble_result = await self.ensemble(solutions=[revised_solution['response'], final_review_result['response']], problem=problem)
        
        # Return the final solution and cost
        return ensemble_result['response'], self.llm.cost_manager.total_cost
