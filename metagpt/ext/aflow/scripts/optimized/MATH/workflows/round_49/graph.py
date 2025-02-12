from typing import Literal
import metagpt.ext.aflow.scripts.optimized.MATH.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.MATH.workflows.round_49.prompt as prompt_custom
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
        self.programmer = operator.Programmer()

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        # Generate initial solution
        response = await self.custom(input=problem, instruction="")
        # Validate the solution using a review step
        review_response = await self.custom(input=response['response'], instruction="Review this solution for correctness.")
        if review_response['response'] != "Correct":
            # If the solution is incorrect, use the Programmer to find the correct answer
            solution_code = f"def solve_problem():\n    return '{problem}'"  # Placeholder for actual logic
            solution_response = await self.programmer(problem=solution_code)
            return solution_response['output'], self.llm.cost_manager.total_cost
        return response['response'], self.llm.cost_manager.total_cost
