from typing import Literal
import metagpt.ext.aflow.scripts.optimized.MATH.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.MATH.workflows.round_120.prompt as prompt_custom
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

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        # Generate initial solution
        solution = await self.custom(input=problem, instruction=prompt_custom.SOLVE_PROMPT)

        # Validate the solution using Python code
        validation_code = f"def validate_solution():\n    # Your validation logic here\n    return {solution['response']}\n"
        validation_result = await self.programmer(problem=validation_code)

        # Review the solution for clarity and correctness
        review_code = f"def review_solution():\n    solution = {solution['response']}\n    # Review logic to check correctness\n    return solution  # Assume it passes for now"
        review_result = await self.programmer(problem=review_code)

        # Return the final solution and cost
        return review_result['output'], self.llm.cost_manager.total_cost
