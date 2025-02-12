from typing import Literal
import metagpt.ext.aflow.scripts.optimized.MATH.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.MATH.workflows.round_126.prompt as prompt_custom
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

        # Verify the initial solution
        verify_instruction = f"Verify the correctness of the following solution: {solution['response']}"
        verify_result = await self.custom(input=verify_instruction, instruction=prompt_custom.VERIFY_PROMPT)

        # Review the solution for correctness
        review_instruction = f"Review the following solution for correctness: {verify_result['response']}"
        review_result = await self.custom(input=review_instruction, instruction=prompt_custom.REVIEW_PROMPT)
        
        # Validate the solution using Python code
        validation_code = f"def validate_solution():\n    # Your validation logic here\n    return {review_result['response']}\n"
        validation_result = await self.programmer(problem=validation_code)
        
        # Use ensemble to improve the final answer selection
        ensemble_result = await self.ensemble(solutions=[solution['response'], validation_result['output']], problem=problem)
        
        # Revise and Optimize the results before the final return
        optimize_instruction = f"Optimize the following solutions to ensure clarity and correctness: {ensemble_result['response']}"
        optimized_result = await self.custom(input=optimize_instruction, instruction=prompt_custom.OPTIMIZE_PROMPT)

        # Return the final solution and cost
        return optimized_result['response'], self.llm.cost_manager.total_cost
